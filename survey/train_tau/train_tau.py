import torch
import torch.nn as nn
from snntorch import surrogate
import yaml

class BetaLIF(nn.Module):
    """
    betaを受け取るLIFニューロン
    """
    def __init__(self,threshold=1.0, reset_mechanism='subtract',spike_grad=surrogate.fast_sigmoid()):
        super(BetaLIF,self).__init__()
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.membrane_potential = 0.0
        self.spike_grad=spike_grad

    def forward(self, input_current,beta):
        """
        beta: 時定数. input_currentと同じサイズ
        input_current: 入力電流
        """
        # Update membrane potential with input current and decay
        self.membrane_potential = beta * self.membrane_potential + input_current

        # Check for spike
        spike = self.fire()  # 大きければ発火
        self.reset_potential(spike)

        return spike,self.membrane_potential
    
    def fire(self):
        mem_shift=self.membrane_potential-self.threshold #ここで一旦thresholdを引いたpotentailを計算
        spike=self.spike_grad(mem_shift) #ここでおそらく0以上のを発火&勾配計算していると思う
        return spike
    
    def reset_potential(self, spike):
        spike = spike.to(torch.bool)
        if self.reset_mechanism == 'subtract':
            self.membrane_potential[spike] -= self.threshold
        elif self.reset_mechanism == 'zero':
            self.membrane_potential[spike] = 0.0

    def init_potential(self):
        """
        timewindowが終わるたびに実行する  
        時系列をシームレスに学習するときはやらない
        が, そんな機会はないと思う(色々めんどいので)
        """
        self.membrane_potential=0.0


class BetaCSNN(nn.Module):
    """
    推論時に時定数betaを指定できるCSNN
    """
    def __init__(self, conf: dict, spike_grad=surrogate.fast_sigmoid()):
        super(BetaCSNN, self).__init__()
        self.in_size = conf["in_size"]
        self.in_channel = conf["in_channel"]
        self.out_channel = conf["out_channel"]
        self.kernel = conf["kernel"]
        self.stride = conf["stride"]
        self.padding = conf["padding"]
        self.pool_type = conf["pool_type"]
        self.is_bn = conf["is_bn"]
        self.mem_reset_type = conf["mem_reset_type"]
        self.mem_threshold = conf["mem_threshold"]


        modules=[
            nn.Conv2d(self.in_channel,self.out_channel,self.kernel,self.stride,self.padding),
            nn.BatchNorm2d(self.out_channel) if self.is_bn else nn.Identity(),
            nn.MaxPool2d(2),
        ]
        self.layers=nn.Sequential(*modules)
        self.betalif = BetaLIF(self.mem_threshold,self.mem_reset_type,spike_grad=spike_grad)

    def forward(self,spikes,betas):
        """
        :param spikes: [timesteps x batch x in_c x h x w]
        :param betas: [timesteps x batch x in_c x h x w]
        :return out_spikes: [timesteps x batch x out_c x h x w]
        :return out_potentials: [timesteps x batch x out_c x h x w]
        """

        T=spikes.shape[0]
        self.betalif.init_potential()

        out_spikes=[]
        out_potentials=[]
        for t in range(T):
            out=self.layers(spikes[t])
            spike,potential=self.betalif(out,betas[t])
            out_spikes.append(spike)
            out_potentials.append(potential)

        out_spikes = torch.stack(out_spikes, dim=0)
        out_potentials = torch.stack(out_potentials, dim=0)
        
        return out_spikes, out_potentials


class CNN(nn.Module):
    def __init__(self, conf: dict):
        super(CNN, self).__init__()
        self.in_size = conf["in_size"]
        self.in_channel = conf["in_channel"]
        self.hiddens = conf["hiddens"]
        self.out_channel = conf["out_channel"]
        self.pool_type = conf["pool_type"]
        self.is_bn = conf["is_bn"]
        self.range_out = conf["range_out"]

        layers = []
        in_channels = self.in_channel

        #>> 中間層ではサイズを変えない >>
        for hidden in self.hiddens:
            layers.append(nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1))
            if self.is_bn:
                layers.append(nn.BatchNorm2d(hidden))
            layers.append(nn.ReLU())
            if self.pool_type == 'avg':
                layers.append(nn.AvgPool2d(kernel_size=3, stride=1,padding=1))
            elif self.pool_type == 'max':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1,padding=1))
            in_channels = hidden
        #>> 中間層ではサイズを変えない >>


        layers.append(nn.Conv2d(in_channels, self.out_channel, kernel_size=3, stride=1, padding=1))
        if self.is_bn:
            layers.append(nn.BatchNorm2d(self.out_channel))
        if self.pool_type == 'avg':
            layers.append(nn.AvgPool2d(2))
        elif self.pool_type == 'max':
            layers.append(nn.MaxPool2d(2))

        layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, spike):
        """
        :param spike : [timestep x batch x c x h x w]
        """
        x=spike[:,:,0]-spike[:,:,1] #positive eventとnegative eventを1次元で表現する (CNNだからこれでいい)
        print("x shape: ",x.shape)
        x=torch.permute(x,dims=(1,0,2,3)) #[timestep x batch x h x w] -> [batch x timestep x h x w] 時間軸をチャンネルと捉える
        
        return self.range_out*self.layers(x)
    


def load_yaml_as_dict(file_path):
    """
    Load a YAML file and return its contents as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML file contents.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':

    from pathlib import Path
    conf=load_yaml_as_dict(Path(__file__).parent/'conf.yml')

    cnn=CNN(conf["cnn"])
    beta_csnn = BetaCSNN(conf["beta-csnn"])

    # Create dummy data
    timesteps = 10
    batch_size = 1
    in_channel = conf["beta-csnn"]["in_channel"]
    out_channel = conf["beta-csnn"]["out_channel"]
    height = conf["beta-csnn"]["in_size"]
    width = conf["beta-csnn"]["in_size"]

    # Random binary spikes
    in_spike = torch.randint(0, 2, (timesteps, batch_size, in_channel, height, width)).float()

    # Beta values (e.g., all ones)
    short_window=conf["archi-config"]["short_window"]
    betas=[]
    for t in range(timesteps):
        if t < short_window:
            padded_spike = torch.cat([torch.zeros(short_window - t, *in_spike.shape[1:]), in_spike[:t]], dim=0)
        else:
            padded_spike = in_spike[t-short_window:t]
        # print("padded_spike shape: ",padded_spike.shape)
        beta = cnn.forward(padded_spike)
        betas.append(beta)
    betas=torch.stack(betas,dim=0)
    
    # beta_csnn.eval()
    out_spikes,out_potentials=beta_csnn(in_spike,betas)
    print(out_spikes)
    print(out_potentials.shape)


    target_spikes=torch.randint(0,2,size=out_spikes.shape)
    loss=torch.mean((out_spikes-target_spikes)**2)
    print(loss)
    loss.backward()