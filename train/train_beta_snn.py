from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import torch
import argparse
import os
from snntorch import functional as SF
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils import load_yaml,load_hdf5,get_minibatch,plot_results,cut_string_end,TERMINAL_WIDTH
from src.beta_csnn import BetaCSNN


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


    #>> phase1. SNNの学習 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print(f"\033[96m{cut_string_end('PHASE1 TRAINING@SNN'+'='*200,TERMINAL_WIDTH)}\033[0m")
    result_table_phase1=[]

    device = torch.device(f"cuda:{args.device}")
    model=BetaCSNN(conf=model_conf,device=device)
    model.to(device)  # ここでモデルをデバイスに移動
    optim=torch.optim.Adam(model.parameters(),lr=train_conf["lr"])
    criterion=SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    datadir=Path(train_conf["datapath"])/"speed_1.0_times" #基準速度のデータpath
    train_files=os.listdir(datadir/"train")
    for ep in range(epoch):
        print(f"epoch: {ep}")
        model.train()
        train_files_e=deepcopy(train_files)
        train_acc_list=[]
        train_loss_list=[]
        for i in tqdm(range(int(len(train_files)/minibatch))):

            minibatch_files,train_files_e=get_minibatch(train_files_e,minibatch)
            data,target=load_hdf5(file_path_list=[
                datadir/"train"/f for f in minibatch_files
            ])

            data=torch.Tensor(np.array(data)).to(device)
            target=torch.Tensor(np.array(target)).to(device)
            data=torch.permute(data,(1,0,2,3,4)) #timestepを一番先頭次元に

            out,_=model.forward(data,is_train_beta=True)
            loss:torch.Tensor=criterion(out,target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss_list+=[loss.item()]
            train_acc_list+=[SF.accuracy_rate(out,target)]

            if len(train_files_e)<minibatch:
                break
            if i>iter and iter>0:
                break
        
        # エポックごとの損失と精度の平均と標準偏差を計算
        train_loss_mean = np.mean(train_loss_list)
        train_loss_std = np.std(train_loss_list)
        train_acc_mean = np.mean(train_acc_list)
        train_acc_std = np.std(train_acc_list)



        #>> test >>
        with torch.no_grad():
            model.eval()
            test_files=os.listdir(datadir/"test")
            test_acc_list=[]
            test_loss_list=[]
            for i in range(int(len(test_files)/minibatch)):
                minibatch_files,test_files=get_minibatch(test_files,minibatch)
                data,target=load_hdf5(file_path_list=[
                    datadir/"test"/f for f in minibatch_files
                ])
                data=torch.Tensor(np.array(data)).to(device)
                target=torch.Tensor(np.array(target)).to(device)
                data=torch.permute(data,(1,0,2,3,4)) #timestepを一番先頭次元に

                out,_=model.forward(data,is_train_beta=False)
                acc=SF.accuracy_rate(out,target)
                test_acc_list+=[acc]
                loss:torch.Tensor=criterion(out,target)
                test_loss_list+=[loss.item()]

                if len(test_files)<minibatch:
                    break
                if i>iter and iter>0:
                    break
                
            test_acc_mean,test_acc_std=np.mean(test_acc_list),np.std(test_acc_list)
            test_loss_mean,test_loss_std=np.mean(test_loss_list),np.std(test_loss_list)
        #>> test >>

        print(f"Train Loss: {train_loss_mean:.2f} ± {train_loss_std:.2f}, Train Accuracy: {train_acc_mean:.2f} ± {train_acc_std:.2f}, Test Loss: {test_loss_mean:.2f} ± {test_loss_std:.2f}, Test Acc: {test_acc_mean:.2f} ± {test_acc_std:.2f}")
        
        #>> 結果の保存 >>
        result_table_phase1+=[[
             ep,train_loss_mean,train_loss_std,train_acc_mean,train_acc_std,
             test_loss_mean,test_loss_std,test_acc_mean,test_acc_std
             ]]
        result_table_db=pd.DataFrame(
            result_table_phase1,
            columns=[
                "ep",
                "train_loss_mean","train_loss_std","train_acc_mean","train_acc_std",
                "test_loss_mean","test_loss_std","test_acc_mean","test_acc_std"
                ])
        result_table_db.to_csv(resultpath/"result_phase1.csv",index=False)
        plot_results(resultpath/"result_phase1.csv",resultpath/"result_phase1.png")
        #>> 結果の保存 >>


        if ep%save_interval==0:
            modelpath=resultpath/"phase1_models"
            if not os.path.exists(modelpath):
                os.makedirs(modelpath)
            torch.save(model.state_dict(),modelpath/f"model_ep{ep}.pth")
    torch.save(model.state_dict(),modelpath/f"model_final.pth")
    print("\033[92mPHASE1 DONE\033[0m")
    #>> phase1. SNNの学習 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



    #>> phase2. CNNの学習 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 全ての速度のデータを与える
    print(f"\033[96m{cut_string_end('PHASE2 TRAINING@CNN'+'='*200,TERMINAL_WIDTH)}\033[0m")
    result_table_phase2=[]

    optim=torch.optim.Adam(model.csnn.parameters(),lr=train_conf["lr"])

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
            target=torch.Tensor(np.array(target)).to(device)
            data=torch.permute(data,(1,0,2,3,4)) #timestepを一番先頭次元に
            # print("data: ",data.shape,"target: ",target.shape)


            out,_=model.forward(data,is_train_beta=True)
            loss:torch.Tensor=criterion(out,target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss_list+=[loss.item()]
            train_acc_list+=[SF.accuracy_rate(out,target)]

            if min([len(files) for files in train_files])<minibatch_j:
                break
            if i>iter and iter>0:
                break

        # エポックごとの損失と精度の平均と標準偏差を計算
        train_loss_mean = np.mean(train_loss_list)
        train_loss_std = np.std(train_loss_list)
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
                target=torch.Tensor(np.array(target)).to(device)
                data=torch.permute(data,(1,0,2,3,4)) #timestepを一番先頭次元に

                out,_=model.forward(data,is_train_beta=True)
                acc=SF.accuracy_rate(out,target)
                test_acc_list+=[acc]
                loss:torch.Tensor=criterion(out,target)
                test_loss_list+=[loss.item()]

                if min([len(files) for files in test_files])<minibatch_j:
                    break
                if i>iter and iter>0:
                    break
                
            test_acc_mean,test_acc_std=np.mean(test_acc_list),np.std(test_acc_list)
            test_loss_mean,test_loss_std=np.mean(test_loss_list),np.std(test_loss_list)
        #>> test >>

        print(f"Train Loss: {train_loss_mean:.2f} ± {train_loss_std:.2f}, Train Accuracy: {train_acc_mean:.2f} ± {train_acc_std:.2f}, Test Loss: {test_loss_mean:.2f} ± {test_loss_std:.2f}, Test Acc: {test_acc_mean:.2f} ± {test_acc_std:.2f}")

        #>> 結果の保存 >>
        result_table_phase2+=[[
             ep,train_loss_mean,train_loss_std,train_acc_mean,train_acc_std,
             test_loss_mean,test_loss_std,test_acc_mean,test_acc_std
             ]]
        result_table_db=pd.DataFrame(
            result_table_phase2,
            columns=[
                "ep",
                "train_loss_mean","train_loss_std","train_acc_mean","train_acc_std",
                "test_loss_mean","test_loss_std","test_acc_mean","test_acc_std"
                ])
        result_table_db.to_csv(resultpath/"result_phase2.csv",index=False)
        plot_results(resultpath/"result_phase2.csv",resultpath/"result_phase2.png")
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

