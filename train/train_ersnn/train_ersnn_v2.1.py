"""
ERSNNv2のモデルを学習するスクリプト
βをスカラにする βをLSTMで推定する

ver2.1
lossにイベント密度によるβ・γ調整関数を追加した
"""


from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import argparse
import os
from snntorch import functional as SF
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pandas as pd

from train.utils import load_yaml,load_hdf5,get_minibatch,cut_string_end,TERMINAL_WIDTH
from src.ersnn_v2 import ERSNNv2


def plot_results(df, result_png_path):
    import matplotlib.pyplot as plt

    # スタイルの設定
    plt.style.use('fivethirtyeight')
    # フォントの設定
    plt.rcParams['font.family'] = 'serif'
    # 背景色の設定
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    
    # Total Loss
    axs[0, 0].plot(df['ep'], df['train_loss_total'], label='Train Total Loss')
    axs[0, 0].fill_between(df['ep'], df['train_loss_total'] - df['train_loss_total_std'], df['train_loss_total'] + df['train_loss_total_std'], alpha=0.2)
    axs[0, 0].plot(df['ep'], df['test_loss_total'], label='Test Total Loss')
    axs[0, 0].fill_between(df['ep'], df['test_loss_total'] - df['test_loss_total_std'], df['test_loss_total'] + df['test_loss_total_std'], alpha=0.2)
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].legend()
    
    # Prediction Loss
    axs[0, 1].plot(df['ep'], df['train_loss_pred'], label='Train Prediction Loss')
    axs[0, 1].fill_between(df['ep'], df['train_loss_pred'] - df['train_loss_pred_std'], df['train_loss_pred'] + df['train_loss_pred_std'], alpha=0.2)
    axs[0, 1].plot(df['ep'], df['test_loss_pred'], label='Test Prediction Loss')
    axs[0, 1].fill_between(df['ep'], df['test_loss_pred'] - df['test_loss_pred_std'], df['test_loss_pred'] + df['test_loss_pred_std'], alpha=0.2)
    axs[0, 1].set_title('Prediction Loss')
    axs[0, 1].legend()
    
    # Beta Loss
    axs[1, 0].plot(df['ep'], df['train_loss_beta'], label='Train Beta Loss')
    axs[1, 0].fill_between(df['ep'], df['train_loss_beta'] - df['train_loss_beta_std'], df['train_loss_beta'] + df['train_loss_beta_std'], alpha=0.2)
    axs[1, 0].plot(df['ep'], df['test_loss_beta'], label='Test Beta Loss')
    axs[1, 0].fill_between(df['ep'], df['test_loss_beta'] - df['test_loss_beta_std'], df['test_loss_beta'] + df['test_loss_beta_std'], alpha=0.2)
    axs[1, 0].set_title('Beta Loss')
    axs[1, 0].legend()
    
    # Gamma Loss
    axs[1, 1].plot(df['ep'], df['train_loss_gamma'], label='Train Gamma Loss')
    axs[1, 1].fill_between(df['ep'], df['train_loss_gamma'] - df['train_loss_gamma_std'], df['train_loss_gamma'] + df['train_loss_gamma_std'], alpha=0.2)
    axs[1, 1].plot(df['ep'], df['test_loss_gamma'], label='Test Gamma Loss')
    axs[1, 1].fill_between(df['ep'], df['test_loss_gamma'] - df['test_loss_gamma_std'], df['test_loss_gamma'] + df['test_loss_gamma_std'], alpha=0.2)
    axs[1, 1].set_title('Gamma Loss')
    axs[1, 1].legend()
    
    for ax in axs.flat:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')

    plt.tight_layout()
    plt.savefig(result_png_path/"loss.png")


    # Plot accuracy in a separate figure
    fig_acc, ax_acc = plt.subplots(figsize=(10, 5))
    ax_acc.plot(df['ep'], df['train_acc'], label='Train Accuracy')
    ax_acc.fill_between(df['ep'], df['train_acc'] - df['train_acc_std'], df['train_acc'] + df['train_acc_std'], alpha=0.2)
    ax_acc.plot(df['ep'], df['test_acc'], label='Test Accuracy')
    ax_acc.fill_between(df['ep'], df['test_acc'] - df['test_acc_std'], df['test_acc'] + df['test_acc_std'], alpha=0.2)
    ax_acc.set_title('Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    ax_acc.spines['top'].set_color('white')
    ax_acc.spines['right'].set_color('white')
    ax_acc.spines['left'].set_color('white')
    ax_acc.spines['bottom'].set_color('white')

    plt.tight_layout()
    plt.savefig(result_png_path/"acc.png")



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
    save_interval=train_conf["save_interval"]
    minibatch=train_conf["batch"]
    device = torch.device(f"cuda:{args.device}")
    model=ERSNNv2(conf=model_conf,device=device)
    model.to(device)  # ここでモデルをデバイスに移動
    criterion=SF.ce_rate_loss() 
    # criterion=SF.mse_count_loss(correct_rate=0.8,incorrect_rate=0.2) #こっちのほうが精度出る気がする


    #>> phase2. CNNの学習 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 全ての速度のデータを与える
    print(f"\033[96m{cut_string_end('PHASE2 TRAINING@CNN'+'='*200,TERMINAL_WIDTH)}\033[0m")
    result_table_phase2=[]

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
        train_acc_list=[]
        train_loss_list=[]
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

            train_loss_list+=[[loss.item(),loss_pred.item(),loss_beta.item(),loss_gamma.item()]]
            train_acc_list+=[SF.accuracy_rate(out,target)]

            if min([len(files) for files in train_files])<minibatch_j:
                break
            if i>iter and iter>0:
                break

        # エポックごとの損失と精度の平均と標準偏差を計算
        train_loss_mean = np.mean(train_loss_list,axis=0)
        train_loss_std = np.std(train_loss_list,axis=0)
        train_acc_mean = np.mean(train_acc_list)
        train_acc_std = np.std(train_acc_list)


        #>> test >>
        with torch.no_grad():
            model.eval()
            test_files=[os.listdir(datadir/"test") for datadir in datadirs]
            test_min_data_length=min([len(files) for files in test_files]) #これを1epochに見るデータサイズとする
            test_acc_list=[]
            test_loss_list=[]
            for i in range(int(test_min_data_length/minibatch_j)):

                #>> 学習データのロード >>
                data,target=[],[]
                for j in range(len(test_files)):
                    minibatch_files,test_files[j]=get_minibatch(test_files[j],minibatch_j)
                    data_j,target_j=load_hdf5(file_path_list=[
                        datadirs[j]/"test"/f for f in minibatch_files
                    ])
                    data+=data_j
                    target+=target_j
                #>> 学習データのロード >>

                data=torch.Tensor(np.array(data)).to(device)
                target=torch.Tensor(np.array(target)).to(device).to(torch.long)
                data=torch.permute(data,(1,0,2,3,4)) #timestepを一番先頭次元に

                out,_,beta,gamma=model.forward(data,is_beta=True,is_gamma=True,return_internal_param=True)
                loss_result=model.loss_func(
                    in_spike=data,out_spike=out,target=target,
                    criterion=criterion,beta=beta,gamma=gamma
                )
                loss_pred,loss_beta,loss_gamma=loss_result["loss_pred"],loss_result["loss_beta"],loss_result["loss_gamma"]
                loss:torch.Tensor=loss_pred+loss_beta+loss_gamma #3種のlossを合計
                
                acc=SF.accuracy_rate(out,target)
                test_acc_list+=[acc]
                loss:torch.Tensor=criterion(out,target)
                test_loss_list+=[[loss.item(),loss_pred.item(),loss_beta.item(),loss_gamma.item()]]

                if min([len(files) for files in test_files])<minibatch_j:
                    break
                if i>iter and iter>0:
                    break
                
            test_acc_mean,test_acc_std=np.mean(test_acc_list),np.std(test_acc_list)
            test_loss_mean,test_loss_std=np.mean(test_loss_list,axis=0),np.std(test_loss_list,axis=0)
        #>> test >>

        print(cut_string_end("="*200,TERMINAL_WIDTH))
        print(f"Train Loss: Total: {train_loss_mean[0]:.2f} ± {train_loss_std[0]:.2f},\n"
              f"            Pred : {train_loss_mean[1]:.2f} ± {train_loss_std[1]:.2f},\n"
              f"            Beta : {train_loss_mean[2]:.2f} ± {train_loss_std[2]:.2f},\n"
              f"            Gamma: {train_loss_mean[3]:.2f} ± {train_loss_std[3]:.2f},\n"
              f"Train Accuracy   : {train_acc_mean:.2f} ± {train_acc_std:.2f},\n"
              "----------------------------"+"\n"
              f"Test Loss: Total: {test_loss_mean[0]:.2f} ± {test_loss_std[0]:.2f},\n"
              f"           Pred : {test_loss_mean[1]:.2f} ± {test_loss_std[1]:.2f},\n"
              f"           Beta : {test_loss_mean[2]:.2f} ± {test_loss_std[2]:.2f},\n"
              f"           Gamma: {test_loss_mean[3]:.2f} ± {test_loss_std[3]:.2f},\n"
              f"Test Accuracy   : {test_acc_mean:.2f} ± {test_acc_std:.2f}")
        print(cut_string_end("="*200,TERMINAL_WIDTH))


        result_table_phase2+=[
            [ep]+train_loss_mean.tolist()+train_loss_std.tolist()+
            [train_acc_mean, train_acc_std]+test_loss_mean.tolist()+test_loss_std.tolist()+
            [test_acc_mean, test_acc_std]
        ]        
        
        result_table_db=pd.DataFrame(
            result_table_phase2,
            columns=[
                "ep",
                "train_loss_total","train_loss_pred","train_loss_beta","train_loss_gamma",
                "train_loss_total_std","train_loss_pred_std","train_loss_beta_std","train_loss_gamma_std",
                "train_acc","train_acc_std",  # 修正: "train_std" を "train_acc_std" に変更
                "test_loss_total","test_loss_pred","test_loss_beta","test_loss_gamma",
                "test_loss_total_std","test_loss_pred_std","test_loss_beta_std","test_loss_gamma_std",
                "test_acc","test_acc_std"
            ]
        )
        
        result_table_db.to_csv(resultpath/"result_phase2.csv",index=False)
        plot_results(result_table_db,resultpath)
        #>> 結果の保存 >>


        if ep%save_interval==0:
            modelpath=resultpath/"phase2_models"
            if not os.path.exists(modelpath):
                os.makedirs(modelpath)
            torch.save(model.state_dict(),modelpath/f"model_ep{ep}.pth")
            
    torch.save(model.state_dict(),modelpath/f"model_final.pth")
    print("\033[92mPHASE2 DONE\033[0m")
    #>> phase2. CNNの学習 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



if __name__ == '__main__':
    main()

