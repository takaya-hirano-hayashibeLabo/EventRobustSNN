import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import torch.nn as nn
import torch


def get_conv_outsize(model,in_size,in_channel):
    input_tensor = torch.randn(1, in_channel, in_size, in_size)
    with torch.no_grad():
        output = model(input_tensor)
    return output.shape


class CSNN(nn.Module):
    def __init__(self, conf:dict,spike_grad=surrogate.fast_sigmoid()):
        super().__init__()
        self.in_size = conf['in_size']
        self.in_channel = conf['in_channel']
        self.out_size = conf['out_size']
        self.hiddens = conf['hiddens']
        self.linear_hidden = conf['linear_hidden']
        self.pool_type = conf['pool_type']
        self.is_bn = conf['is_bn']
        self.beta = conf['beta']
        self.mem_threshold = conf['mem_threshold']
        self.dropout_rate=conf['dropout_rate']


        modules=[]

        #>> CSNNの構築 >>
        in_channel=self.in_channel
        for hidden in self.hiddens:
            modules+=[
                nn.Conv2d(in_channels=in_channel,out_channels=hidden,kernel_size=3,stride=1,padding=1),
            ]

            if self.is_bn:
                modules+=[
                    nn.BatchNorm2d(hidden),
                ]

            if self.pool_type=='avg'.casefold():
                modules+=[
                    nn.AvgPool2d(2),
                ]
            elif self.pool_type=="max".casefold():
                modules+=[
                    nn.MaxPool2d(2),
                ]

            modules+=[
                snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True,threshold=self.mem_threshold),
                nn.Dropout(self.dropout_rate),
            ]

            in_channel=hidden
        #>> CSNNの構築 >>


        #>> 線形層の構築 >>
        conv_out=get_conv_outsize(
            nn.Sequential(*modules),
            self.in_size,
            self.in_channel
        )

        modules+=[
            nn.Flatten(),
            nn.Linear(conv_out[1]*conv_out[2]*conv_out[3],self.linear_hidden),
            snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True,threshold=self.mem_threshold),
            nn.Linear(self.linear_hidden,self.out_size),
            snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True,threshold=self.mem_threshold,output=True)
        ]
        #>> 線形層の構築 >>

        self.model=nn.Sequential(*modules)


    def forward(self,spike:torch.Tensor):
        """
        :param spike: [timestep x batch x c x h x w]
        :return out_spike: [timestep x batch x out_size]
        :return out_mem: [timestep x batch x out_size]
        """

        utils.reset(self.model)

        T=spike.shape[0]
        out_spike=[]
        out_mem=[]
        for t in range(T):
            sp,mem=self.model(spike[t])
            out_spike.append(sp)
            out_mem.append(mem)

        return torch.stack(out_spike),torch.stack(out_mem)


if __name__=="__main__":
    conf = {
        "in_size": 32,
        "in_channel": 2,
        "out_size": 10,
        "hiddens": [32, 64],
        "linear_hidden": 256,
        "pool_type": "avg",
        "is_bn": True,
        "beta": 0.5,
        "mem_threshold": 1.0,
    }

    model=CSNN(conf)

    # テスト用のデータを作成
    test_data = torch.randn(100, 256, conf['in_channel'], conf['in_size'], conf['in_size'])  # [timestep x batch x c x h x w]

    # モデルにデータを流す
    out_spike, out_mem = model(test_data)

    # 結果を出力
    print("Output spikes shape:", out_spike.shape)
    print("Output membrane potentials shape:", out_mem.shape)