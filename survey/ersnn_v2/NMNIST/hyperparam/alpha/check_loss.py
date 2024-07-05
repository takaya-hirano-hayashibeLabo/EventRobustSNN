# とりあえず適当なデータ流したときのlossβとlossγをしらべる

"""
ERSNNv2のモデルを学習するスクリプト
βをスカラにする βをLSTMで推定する

ver2.1
lossにイベント密度によるβ・γ調整関数を追加した
"""


from pathlib import Path
import sys
ROOT=Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(ROOT)) #root

import torch
import argparse
import os
from snntorch import functional as SF
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pandas as pd

from train.utils import load_yaml,load_hdf5,get_minibatch,plot_results,cut_string_end,TERMINAL_WIDTH
from src.ersnn_v2 import ERSNNv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str,help="configのあるパス")
    parser.add_argument("--device",default=0,help="GPUの番号")
    args = parser.parse_args()

    resultpath=Path(args.target)/"result"
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    conf=load_yaml(Path(args.target)/"conf.yml")
    train_conf,model_conf=conf["train"],conf["model"]


    epoch=train_conf["epoch"]
    iter=train_conf["iter"]
    minibatch=train_conf["batch"]
    device = torch.device(f"cuda:{args.device}")
    model=ERSNNv2(conf=model_conf,device=device)
    model.to(device)  # ここでモデルをデバイスに移動
    criterion=SF.ce_rate_loss() 
    # criterion=SF.mse_count_loss(correct_rate=0.8,incorrect_rate=0.2) #こっちのほうが精度出る気がする


    #>> phase2. CNNの学習 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 全ての速度のデータを与える
    print(f"\033[96m{cut_string_end('PHASE2 TRAINING@CNN'+'='*200,TERMINAL_WIDTH)}\033[0m")

    optim=torch.optim.Adam(
        model.parameters(),
        lr=train_conf["lr"]
        )

    datadirs=[Path(train_conf["datapath"])/file_name for file_name in os.listdir(Path(train_conf["datapath"]))] #速度ごとのデータpath
    train_files=[os.listdir(datadir/"train") for datadir in datadirs]
    min_data_length=min([len(files) for files in train_files]) #これを1epochに見るデータサイズとする
    minibatch_j=int(minibatch/len(datadirs)) #各speedのminibatch. 全部のdirからminibatchとるとメモリが足りなくなる

    for ep in range(epoch):
        print(f"epoch: {ep}")
        model.train()
        train_files_epoch=deepcopy(train_files)
        loss_list=[]
        for i in tqdm(range(int(min_data_length/minibatch_j))):


            #>> 学習データのロード >>
            data,target=[],[]
            for j in range(len(train_files_epoch)):
                minibatch_files,train_files_epoch[j]=get_minibatch(train_files_epoch[j],minibatch_j)
                data_j,target_j=load_hdf5(file_path_list=[
                    datadirs[j]/"train"/f for f in minibatch_files
                ])
                data+=data_j
                target+=target_j
            #>> 学習データのロード >>
            
            
            data=torch.Tensor(np.array(data)).to(device)
            target=torch.Tensor(np.array(target)).to(device).to(torch.long) #crossentropyにするならラベルはlong
            data=torch.permute(data,(1,0,2,3,4)) #timestepを一番先頭次元に
            # print("data: ",data.shape, data.dtype,"target: ",target.shape,target.dtype)
            # print(target)


            out,_,beta,gamma=model.forward(data,is_beta=True,is_gamma=True,return_internal_param=True)
            loss_result=model.loss_func(
                in_spike=data,out_spike=out,target=target,
                criterion=criterion,beta=beta,gamma=gamma
            )
            loss_pred,loss_beta,loss_gamma=loss_result["loss_pred"],loss_result["loss_beta"],loss_result["loss_gamma"]
            loss:torch.Tensor=loss_pred+loss_beta+loss_gamma #3種のlossを合計

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_list.append(
                [[loss_pred.item(),loss_beta.item(),loss_gamma.item()]]
            )

            if min([len(files) for files in train_files])<minibatch_j:
                break
            if i>iter and iter>0:
                break

    #>> phase2. CNNの学習 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # 学習ループの最後にloss_listをプロットして保存
    import matplotlib.pyplot as plt
    loss_array = np.array(loss_list).squeeze()
    fig, axs = plt.subplots(3, 1, figsize=(8, 9))

    axs[0].plot(loss_array[:, 0], label='Loss Pred')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss Pred')
    axs[0].legend()
    axs[0].set_title('Prediction Loss')

    axs[1].plot(loss_array[:, 1], label='Loss Beta')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Loss Beta')
    axs[1].legend()
    axs[1].set_title('Beta Loss')

    axs[2].plot(loss_array[:, 2], label='Loss Gamma')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Loss Gamma')
    axs[2].legend()
    axs[2].set_title('Gamma Loss')

    plt.tight_layout()
    plt.savefig(resultpath / 'loss_plot.png')
    plt.close()


if __name__ == '__main__':
    main()

