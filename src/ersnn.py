"""
時定数だけでなく膜抵抗も可変にしたSNN
これによって、空間パターンにも強くなると思われる
"""

import torch
import torch.nn as nn
from snntorch import surrogate
from .csnn import CSNN


class ERLIF(nn.Module):
    """
    EventRobustLIFニューロン
    時定数βと膜抵抗γを受け取る
    """
    def __init__(self,threshold=1.0, reset_mechanism='subtract',spike_grad=surrogate.fast_sigmoid()):
        super(ERLIF,self).__init__()
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.membrane_potential = 0.0
        self.spike_grad=spike_grad

    def forward(self, input_current,beta,gamma):
        """
        beta: 時定数. input_currentと同じサイズ
        gamma: 膜抵抗 input_currentと同じサイズ
        input_current: 入力電流
        """
        # Update membrane potential with input current and decay
        # betaによって時間的な記憶量を調整. gammaによって空間的な記憶量を調整
        self.membrane_potential = beta * self.membrane_potential + gamma * input_current

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
        全体のtimewindowが終わるたびに実行する  
        時系列をシームレスに学習するときはやらない
        が, そんな機会はないと思う(色々めんどいので)
        """
        self.membrane_potential=0.0



class ERSNN(nn.Module):
    """
    時定数betaを調整しながら推論するCSNN
    """
    def __init__(self, conf: dict, spike_grad=surrogate.fast_sigmoid(),device="cpu"):
        super(ERSNN, self).__init__()
        self.device=device


        #>> beta-CNNの構築 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        beta_cnn_conf=conf["beta-cnn"]
        self.beta_cnn_in_size = beta_cnn_conf["in_size"]
        self.beta_cnn_in_channel = beta_cnn_conf["in_channel"]
        self.beta_cnn_hiddens = beta_cnn_conf["hiddens"]
        self.beta_cnn_out_channel = beta_cnn_conf["out_channel"]
        self.beta_cnn_pool_type = beta_cnn_conf["pool_type"]
        self.beta_cnn_is_bn = beta_cnn_conf["is_bn"]
        self.beta_cnn_range_out = beta_cnn_conf["range_out"]
        self.beta_cnn_window=beta_cnn_conf["window"]
        self.beta_cnn_dropout_rate=beta_cnn_conf["dropout_rate"]

        layers = []
        in_channels = self.beta_cnn_in_channel

        ## >> 中間層ではサイズを変えない >>
        for hidden in self.beta_cnn_hiddens:
            layers.append(nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1))
            if self.beta_cnn_is_bn:
                layers.append(nn.BatchNorm2d(hidden))
            if self.beta_cnn_pool_type == 'avg':
                layers.append(nn.AvgPool2d(kernel_size=3, stride=1,padding=1))
            elif self.beta_cnn_pool_type == 'max':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1,padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.beta_cnn_dropout_rate))

            in_channels = hidden
        ## >> 中間層ではサイズを変えない >>


        layers.append(nn.Conv2d(in_channels, self.beta_cnn_out_channel, kernel_size=3, stride=1, padding=1))
        if self.beta_cnn_is_bn:
            layers.append(nn.BatchNorm2d(self.beta_cnn_out_channel))
        if self.beta_cnn_pool_type == 'avg':
            layers.append(nn.AvgPool2d(2))
        elif self.beta_cnn_pool_type == 'max':
            layers.append(nn.MaxPool2d(2))

        layers.append(nn.Tanh())

        self.beta_cnn = nn.Sequential(*layers)
        #<< beta-CNNの構築 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> gamma-CNNの構築 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        gamma_cnn_conf=conf["gamma-cnn"]
        self.gamma_cnn_in_size = gamma_cnn_conf["in_size"]
        self.gamma_cnn_in_channel = gamma_cnn_conf["in_channel"]
        self.gamma_cnn_hiddens = gamma_cnn_conf["hiddens"]
        self.gamma_cnn_out_channel = gamma_cnn_conf["out_channel"]
        self.gamma_cnn_pool_type = gamma_cnn_conf["pool_type"]
        self.gamma_cnn_is_bn = gamma_cnn_conf["is_bn"]
        self.gamma_cnn_dropout_rate=gamma_cnn_conf["dropout_rate"]

        layers = []
        in_channels = self.gamma_cnn_in_channel

        ## >> 中間層ではサイズを変えない >>
        for hidden in self.gamma_cnn_hiddens:
            layers.append(nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1))
            if self.gamma_cnn_is_bn:
                layers.append(nn.BatchNorm2d(hidden))
            if self.gamma_cnn_pool_type == 'avg':
                layers.append(nn.AvgPool2d(kernel_size=3, stride=1,padding=1))
            elif self.gamma_cnn_pool_type == 'max':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1,padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.gamma_cnn_dropout_rate))

            in_channels = hidden
        ## >> 中間層ではサイズを変えない >>


        layers.append(nn.Conv2d(in_channels, self.gamma_cnn_out_channel, kernel_size=3, stride=1, padding=1))
        if self.gamma_cnn_is_bn:
            layers.append(nn.BatchNorm2d(self.gamma_cnn_out_channel))
        if self.gamma_cnn_pool_type == 'avg':
            layers.append(nn.AvgPool2d(2))
        elif self.gamma_cnn_pool_type == 'max':
            layers.append(nn.MaxPool2d(2))

        layers.append(nn.Sigmoid()) #0~1で情報を選択する

        self.gamma_cnn = nn.Sequential(*layers)
        #<< gamma-CNNの構築 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        



        #>> beta-CSNNの構築 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        snn_conf = conf["ersnn"]
        self.snn_in_size = snn_conf["in_size"]
        self.snn_in_channel = snn_conf["in_channel"]
        self.snn_out_channel = snn_conf["out_channel"]
        self.snn_kernel = snn_conf["kernel"]
        self.snn_stride = snn_conf["stride"]
        self.snn_padding = snn_conf["padding"]
        self.snn_pool_type = snn_conf["pool_type"]
        self.snn_is_bn = snn_conf["is_bn"]
        self.snn_mem_reset_type = snn_conf["mem_reset_type"]
        self.snn_mem_threshold = snn_conf["mem_threshold"]
        self.snn_init_beta=snn_conf["init_beta"]
        self.snn_beta_size=snn_conf["beta_size"]
        self.snn_gamma_size=snn_conf["gamma_size"]

        modules = [
            nn.Conv2d(self.snn_in_channel, self.snn_out_channel, self.snn_kernel, self.snn_stride, self.snn_padding),
            nn.BatchNorm2d(self.snn_out_channel) if self.snn_is_bn else nn.Identity(),
            nn.MaxPool2d(2),
        ]
        self.snn_layers = nn.Sequential(*modules)
        self.beta_lif = ERLIF(self.snn_mem_threshold, self.snn_mem_reset_type, spike_grad=spike_grad)
        #<< beta-CSNNの構築 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        #>> 推論を行うCSNNの構築 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.csnn=CSNN(conf=conf["snn"],spike_grad=spike_grad)
        #<< 推論を行うCSNNの構築 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    def forward(self,spike,is_beta=False,is_gamma=False):
        """
        :param spike: [timesteps x batch x in_c x in_h x in_w]
        :param is_beta: beta調整するかどうか
        :param is_gamma: gamma調整するかどうか
        :return out_spikes: [timesteps x batch x out_c x out_h x out_w]
        :return out_potentials: [timesteps x batch x out_c x out_h x out_w]
        """

        T,batch,_,_,_=spike.shape #spikeの全体時間とバッチサイズ
        # print("spike shape: ",spike.shape)


        #>> CNNによるbetaの推論 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if is_beta:
            betas=[]

            # CNNに入力するためにspikeの形式を変換する
            # positive eventとnegative eventを1次元で表現する
            # この式によって, 00->-1, 10->0.5, 01->-05, 11->1の4状態に連続な式で変換できる
            x=1.5*spike[:,:,0]+0.5*spike[:,:,1]-1 
       
            for t in range(T):
                if t < self.beta_cnn_window:
                    padded_x = torch.cat([torch.zeros(self.beta_cnn_window - t, *x.shape[1:]).to(self.device), x[:t]], dim=0)
                else:
                    padded_x = x[t-self.beta_cnn_window:t]
                padded_x=torch.permute(padded_x,dims=(1,0,2,3)) #[timestep x batch x h x w] -> [batch x timestep x h x w] 時間軸をチャンネルと捉える
                beta = self.beta_cnn_range_out*self.beta_cnn.forward(padded_x)
                betas.append(beta)
            betas=self.snn_init_beta+torch.stack(betas,dim=0) #推論したbetaの増分と基本betaを足し合わせる

        elif not is_beta:
            betas=self.snn_init_beta+torch.zeros(T,batch,*self.snn_beta_size) #学習しないときは初期値のまま
        #<< CNNによるbetaの推論 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> CNNによるgammaの推論 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if is_gamma:
            gammas=[]

            for t in range(T):
                gamma=self.gamma_cnn.forward(spike[t]) #その時間だけ見ればいい
                gammas.append(gamma)
            gammas=torch.stack(gammas,dim=0) #[T x batch x out_c x out_h x out_w]

        elif not is_gamma:
            gammas=torch.ones(T,batch,*self.snn_gamma_size)
        #<< CNNによるgammaの推論 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> beta-SNNによるspikeの制御 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # print("beta mean", torch.mean(betas))
        self.beta_lif.init_potential()
        out_spikes=[]
        out_potentials=[]
        for t in range(T):
            # print("spike shape: ",spike[t].shape)
            out=self.snn_layers(spike[t])
            # print("out shape: ",out.shape)
            # print("beta shape: ",betas[t].shape)
            # print("gamma shape: ",gammas[t].shape," type: ",type(gammas[t]))
            sp,potential=self.beta_lif(out,betas[t].to(self.device),gammas[t].to(self.device))
            out_spikes.append(sp)
            out_potentials.append(potential)

        out_spikes = torch.stack(out_spikes, dim=0)
        out_potentials = torch.stack(out_potentials, dim=0)
        #<< beta-SNNによるspikeの制御 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> csnnによる最終的な推論 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        fin_spikes,fin_potentials=self.csnn(out_spikes)
        #<< csnnによる最終的な推論 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        
        return fin_spikes,fin_potentials



    def get_internal_params(self,spike):
        """
        :param spike: [timesteps x batch x in_c x in_h x in_w]
        :return betas: [timesteps x batch x out_c x out_h x out_w]
        :return gammas: [timesteps x batch x out_c x out_h x out_w]
        """

        T,batch,_,_,_=spike.shape #spikeの全体時間とバッチサイズ
        # print("spike shape: ",spike.shape)


        #>> CNNによるbetaの推論 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        betas=[]

        # CNNに入力するためにspikeの形式を変換する
        # positive eventとnegative eventを1次元で表現する
        # この式によって, 00->-1, 10->0.5, 01->-05, 11->1の4状態に連続な式で変換できる
        x=1.5*spike[:,:,0]+0.5*spike[:,:,1]-1 
    
        for t in range(T):
            if t < self.beta_cnn_window:
                padded_x = torch.cat([torch.zeros(self.beta_cnn_window - t, *x.shape[1:]).to(self.device), x[:t]], dim=0)
            else:
                padded_x = x[t-self.beta_cnn_window:t]
            padded_x=torch.permute(padded_x,dims=(1,0,2,3)) #[timestep x batch x h x w] -> [batch x timestep x h x w] 時間軸をチャンネルと捉える
            beta = self.beta_cnn_range_out*self.beta_cnn.forward(padded_x)
            betas.append(beta)
        betas=self.snn_init_beta+torch.stack(betas,dim=0) #推論したbetaの増分と基本betaを足し合わせる
        #<< CNNによるbetaの推論 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> CNNによるgammaの推論 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        gammas=[]

        for t in range(T):
            gamma=self.gamma_cnn.forward(spike[t]) #その時間だけ見ればいい
            gammas.append(gamma)
        gammas=torch.stack(gammas,dim=0) #[T x batch x out_c x out_h x out_w]
        #<< CNNによるgammaの推論 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        
        return betas,gammas


