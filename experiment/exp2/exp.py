from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import tonic
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tonic.transforms as transforms
import torch
import torchvision
from snntorch import functional as SF


from utils import change_speed_v2,event2anim,get_random_subset,save_args,ListNMNIST,load_yaml,save_results,calculate_accuracy
from src.beta_csnn import BetaCSNN
from src.csnn import CSNN
from src.crnn import CLSTM
from src.ersnn import ERSNN 


class RandomBatchSampler:
    def __init__(self, dataset, batch_size):
        import random
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        random.shuffle(self.indices)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size

        batch = [self.dataset[i] for i in batch_indices]
        inputs, targets = zip(*batch)

        inputs = self.pad_stack(inputs)
        targets = np.array(targets)

        return inputs, targets

    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def pad_stack(self,inputs):
        """
        確かに末尾に0paddingされている  
        1フレーム目と最終フレーム目の平均をとり確認
        1フレーム目は0より大きく, 最終フレームは0だった
        """
        # 各入力の1次元目のサイズを取得
        sizes = [input.shape[0] for input in inputs]
        max_size = max(sizes)
        
        # パディングを行う
        padded_inputs = []
        for input in inputs:
            pad_width = [(0, max_size - input.shape[0])] + [(0, 0)] * (input.ndim - 1)
            padded_input = np.pad(input, pad_width, mode='constant', constant_values=0)
            padded_inputs.append(padded_input)
        
        # スタックする
        return np.stack(padded_inputs)

def plot_statistics(speed_list,mean, std, savepath,file_name):
    import matplotlib
    # スタイルの設定
    plt.style.use('fivethirtyeight')

    # フォントの設定
    plt.rcParams['font.family'] = 'serif'
    # 背景色の設定
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.figure(figsize=(10, 6))
    cmap = matplotlib.colormaps.get_cmap('viridis')

    plt.errorbar(speed_list, mean, yerr=std, fmt='-o', capsize=5, label=f'{file_name} Mean with Std', color=cmap(2/3))
    plt.xlabel("Speed Index")
    plt.ylabel(f"{file_name} Value")
    plt.title(f"{file_name} Mean and Standard Deviation Over Speeds")
    plt.legend()
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=2)

    # レイアウトの自動調整
    plt.tight_layout()
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['bottom'].set_color('white')

    plt.savefig(savepath / f"{file_name}.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str,help="configのあるパス",required=True)
    parser.add_argument("--device",default=0,help="GPUの番号")
    parser.add_argument("--speed_max",default=1,type=float)
    parser.add_argument("--speed_min",default=0.1,type=float)
    parser.add_argument("--speed_step",default=0.1,type=float)
    parser.add_argument("--savepath", type=str, help="保存するディレクトリ",default=str(Path(__file__).parent))
    args=parser.parse_args()

    savepath=Path(args.savepath)/f"result"
    if not savepath.exists():
        os.makedirs(savepath)
    save_args(args,savepath)



    #>> モデルの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device = torch.device(f"cuda:{args.device}")
    model_dirs=os.listdir(args.target)
    models={}
    for dir_name in model_dirs:

        if "ersnn".casefold() in dir_name:
            config=load_yaml(str(Path(args.target)/dir_name/"conf.yml"))["model"]
            beta_snn=ERSNN(config,device=device).to(device)
            beta_snn.load_state_dict(torch.load(str(Path(args.target)/dir_name/"result/phase2_models/model_final.pth")))
            beta_snn.eval()
            models["ersnn"]={"model":beta_snn}
            models["ersnn"]["beta_mean"]=[]
            models["ersnn"]["beta_std"]=[]
            models["ersnn"]["gamma_mean"]=[]
            models["ersnn"]["gamma_std"]=[]

        if "snn".casefold() in dir_name and "beta".casefold() in dir_name:
            config=load_yaml(str(Path(args.target)/dir_name/"conf.yml"))["model"]
            beta_snn=BetaCSNN(config,device=device).to(device)
            beta_snn.load_state_dict(torch.load(str(Path(args.target)/dir_name/"result/phase2_models/model_final.pth")))
            beta_snn.eval()
            models["beta-snn"]={"model":beta_snn}
            models["beta-snn"]["beta_mean"]=[]
            models["beta-snn"]["beta_std"]=[]

    #<< モデルの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    for s in np.arange(args.speed_min,args.speed_max,args.speed_step):

        #>> テストデータの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        datapath=Path(__file__).parent.parent.parent/"original_data"
        testset = tonic.datasets.NMNIST(save_to=str(datapath), train=False)
        exp_event=testset[0]
        testset=get_random_subset(testset,0.1) #データが多すぎるので間引く


        sensor_size = tonic.datasets.NMNIST.sensor_size
        transform=transforms.Compose([
                transforms.Denoise(filter_time=10000),
                transforms.ToFrame(sensor_size=sensor_size, time_window=1400), #time_window msごとのデータをフレームデータに変換する
                torch.from_numpy,
                torchvision.transforms.Resize((32, 32),antialias=True)
            ])


        # #>> 例の動画の保存 >>
        params=[{"speed":s,"rate":1.0}]
        new_events=change_speed_v2(exp_event[0],params)
        # event2anim(exp_event[0],1400,34,34,output_path=savepath/"videos",file_name=f"event{exp_event[1]}_original")
        # event2anim(new_events,1400,34,34,   output_path=savepath/"videos",file_name=f"event{exp_event[1]}_changed")
        # #<< 例の動画の保存 <<


        print(f"\nchanging test event speed to x{s:.2f}...")
        test_data=[]
        for i,(events,target) in tqdm(enumerate(testset),total=len(testset)):
            new_events=change_speed_v2(events,params)
            test_data+=[{"events":new_events,"target":target}]

            if i>50:
                break
        testset=ListNMNIST(test_data,transform=transform)
        test_sampler=RandomBatchSampler(testset,128)
        print(f"\033[92mdone\033[0m frame size : {len(testset[0][0])}")
        #<< テストデータの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> テスト >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        print("\nevaluating models...")
        with torch.no_grad():
            beta_list=[]
            gamma_list=[]
            for i, (inputs, target) in tqdm(enumerate(test_sampler),total=len(test_sampler)):
                
                inputs[inputs==2]=1 #重なって2になってるとこは1にする
                inputs=torch.Tensor(inputs).to(device)
                inputs=torch.permute(inputs,dims=(1,0,2,3,4))
                target=torch.Tensor(target).to(device)
                # print("inputs:",inputs.shape,"targets:",target.shape)
                # print(torch.mean(inputs[0]),torch.mean(inputs[-1])) #0paddingの確認


                for key,item in models.items():
                    if key=="ersnn":
                        betas,gammas=item["model"].get_internal_params(inputs)
                        beta_list.append(betas)
                        gamma_list.append(gammas)

                    if key=="beta-snn":
                        betas=item["model"].get_internal_params(inputs)
                        beta_list.append(betas)

            for key,item in models.items():
                if key=="ersnn":
                    beta_list=torch.stack(beta_list,dim=1) #batch方向にconcat
                    beta_list=torch.mean(beta_list,dim=0) #timestep方向に平均
                    gamma_list=torch.stack(gamma_list,dim=1) #batch方向にconcat
                    gamma_list=torch.mean(gamma_list,dim=0) #timestep方向に平均
                    item["beta_mean"]+=[torch.mean(beta_list).item()]
                    item["beta_std"]+=[torch.std(beta_list).item()]
                    item["gamma_mean"]+=[torch.mean(gamma_list).item()]
                    item["gamma_std"]+=[torch.std(gamma_list).item()]

                if key=="beta-snn":
                    beta_list=torch.stack(beta_list,dim=1) #batch方向にconcat
                    beta_list=torch.mean(beta_list,dim=0) #timestep方向に平均
                    item["beta_mean"]+=[torch.mean(beta_list).item()]
                    item["beta_std"]+=[torch.std(beta_list).item()]
        print("\033[92mdone\033[0m")
        #<< テスト <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    if "ersnn" in models.keys():
        plot_statistics(
            np.arange(args.speed_min,args.speed_max,args.speed_step),
            models["ersnn"]["beta_mean"],
            models["ersnn"]["beta_std"],
            savepath,
            "ersnn_beta"
        )
        plot_statistics(
            np.arange(args.speed_min,args.speed_max,args.speed_step),
            models["ersnn"]["gamma_mean"],
            models["ersnn"]["gamma_std"],
            savepath,
            "ersnn_gamma"
        )

    if "beta-snn" in models.keys():
        plot_statistics(
            np.arange(args.speed_min,args.speed_max,args.speed_step),
            models["beta-snn"]["beta_mean"],
            models["beta-snn"]["beta_std"],
            savepath,
            "beta-snn_beta"
        )


if __name__=="__main__":
    main()