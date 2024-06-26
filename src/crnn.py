
import torch.nn as nn
import torch


def get_conv_outsize(model,in_size,in_channel):
    input_tensor = torch.randn(1, in_channel, in_size, in_size)
    with torch.no_grad():
        output = model(input_tensor)
    return output.shape


class CRNN(nn.Module):
    def __init__(self, conf:dict):
        super().__init__()
        self.in_size = conf['in_size']
        self.in_channel = conf['in_channel']
        self.out_size = conf['out_size']
        self.hiddens = conf['hiddens']
        self.linear_hidden = conf['linear_hidden']
        self.pool_type = conf['pool_type']
        self.is_bn = conf['is_bn']
        self.dropout_rate=conf['dropout_rate']


        modules=[]

        #>> CRNNの構築 >>
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
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
            ]

            in_channel=hidden

        self.cnn=nn.Sequential(*modules) #2次元データの特徴をとるCNNの構築
        #>> CRNNの構築 >>


        #>> リカレント層 >>
        conv_outsize=get_conv_outsize(
            self.cnn,
            self.in_size,
            self.in_channel
        )        
        self.rnn=nn.RNN(
            input_size=conv_outsize[1]*conv_outsize[2]*conv_outsize[3],
            hidden_size=self.linear_hidden,
            num_layers=self.linear_num,
            )
        #>> リカレント層 >>


        #>> 線形層の構築 >>
        self.out_layer=nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.linear_hidden,self.out_size),
        )
        #>> 線形層の構築 >>



    def forward(self,spike:torch.Tensor):
        """
        :param spike: [timestep x batch x c x h x w]
        :return out_spike: [timestep x batch x out_size]
        :return out_mem: [timestep x batch x out_size]
        """

        T=spike.shape[0]
        out=[]
        for t in range(T):
            y=self.cnn(spike[t])
            y=torch.flatten(y,start_dim=1)
            out.append(y)
        out=torch.stack(out)
        out, h_n = self.rnn(out)  # RNNの出力を受け取る (LSTMのc_nは削除)
        out = out[-1]  # 最終stepだけ抽出
        out=self.out_layer(out)
        
        return out
    


class CLSTM(nn.Module):
    def __init__(self, conf:dict):
        super().__init__()
        self.in_size = conf['in_size']
        self.in_channel = conf['in_channel']
        self.out_size = conf['out_size']
        self.hiddens = conf['hiddens']
        self.linear_hidden = conf['linear_hidden']
        self.linear_num=conf['linear_num'] #リカレント層を何層にするか
        self.pool_type = conf['pool_type']
        self.is_bn = conf['is_bn']
        self.dropout_rate=conf['dropout_rate']


        modules=[]

        #>> CRNNの構築 >>
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
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
            ]

            in_channel=hidden

        self.cnn=nn.Sequential(*modules) #2次元データの特徴をとるCNNの構築
        #>> CRNNの構築 >>


        #>> リカレント層 >>
        conv_outsize=get_conv_outsize(
            self.cnn,
            self.in_size,
            self.in_channel
        )        
        self.rnn=nn.LSTM(
            input_size=conv_outsize[1]*conv_outsize[2]*conv_outsize[3],
            hidden_size=self.linear_hidden,
            num_layers=self.linear_num,
            )
        #>> リカレント層 >>


        #>> 線形層の構築 >>
        self.out_layer=nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.linear_hidden,self.out_size),
        )
        #>> 線形層の構築 >>



    def forward(self,spike:torch.Tensor):
        """
        :param spike: [timestep x batch x c x h x w]
        :return out_spike: [timestep x batch x out_size]
        :return out_mem: [timestep x batch x out_size]
        """

        T=spike.shape[0]
        out=[]
        for t in range(T):
            y=self.cnn(spike[t])
            y=torch.flatten(y,start_dim=1)
            out.append(y)
        out=torch.stack(out)
        out, (h_n, c_n) = self.rnn(out)  # LSTMの出力を受け取る
        out = out[-1]  # 最終stepだけ抽出
        out=self.out_layer(out)
        
        return out



if __name__ == "__main__":
    conf = {
        "in_size": 32,
        "in_channel": 2,
        "out_size": 10,
        "hiddens": [16, 32, 64],
        "linear_hidden": 256,
        "pool_type": "avg",
        "is_bn": True,
    }

    model = CRNN(conf)

    # テストデータの作成
    batch_size = 256
    timesteps = 100
    test_input = torch.randn(timesteps, batch_size, conf['in_channel'], conf['in_size'], conf['in_size'])

    # モデルにテストデータを流す
    output = model(test_input)
    print("Output shape:", output.shape)  # Output shape: [timestep, batch, out_size]