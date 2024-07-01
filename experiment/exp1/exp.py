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
from src.ersnn_v2 import ERSNNv2


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

def plot_speed_trj(speed_trj, filepath="speed_trj.png"):
    # スタイルの設定
    plt.style.use('fivethirtyeight')
    # フォントの設定
    plt.rcParams['font.family'] = 'serif'
    # 背景色の設定
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    plt.figure(figsize=(10, 5))
    plt.plot(speed_trj, label="Speed Trajectory")
    plt.xlabel("Event Index")
    plt.ylabel("Speed")
    plt.title("Speed Trajectory Over Events")
    plt.legend()
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=2)


    plt.tight_layout()
    # 現在のAxesオブジェクトを取得
    ax = plt.gca()
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')


    plt.savefig(filepath)
    plt.close()


def generate_speed_trj(event_num,speed_list):
    speed_trj=np.zeros(event_num)
    for i,speed in enumerate(speed_list):
        step=floor(event_num/len(speed_list))

        if i==len(speed_list)-1:
            speed_trj[i*step:]=speed
        else:
            speed_trj[i*step:(i+1)*step]=speed
    return speed_trj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str,help="configのあるパス",required=True)
    parser.add_argument("--device",default=0,help="GPUの番号")
    parser.add_argument("--speed", nargs='+', type=float, help="速度のリスト",required=True)  # リストとして受け取る    args = parser.parse_args()
    parser.add_argument("--rate", nargs='+', type=float, help="速度比のリスト",required=True)  # リストとして受け取る    args = parser.parse_args()
    parser.add_argument("--savepath", type=str, help="保存するディレクトリ",required=True)
    args=parser.parse_args()

    speeds_str = "-".join(f"{speed:.2f}" for speed in args.speed)
    rate_str = "-".join(f"{rate:.2f}" for rate in args.rate)
    savepath=Path(args.savepath)/f"speed{speeds_str}_rate{rate_str}"
    if not savepath.exists():
        os.makedirs(savepath)
    save_args(args,savepath)


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
    params=[]
    for s,r in zip(args.speed,args.rate):
        params+=[
            {"speed":s, "rate":r}
        ]
    new_events=change_speed_v2(exp_event[0],params)
    event2anim(exp_event[0],1400,34,34,output_path=savepath/"videos",file_name=f"event{exp_event[1]}_original")
    event2anim(new_events,1400,34,34,   output_path=savepath/"videos",file_name=f"event{exp_event[1]}_changed")
    # #<< 例の動画の保存 <<


    print("\nchanging test event speed...")
    test_data=[]
    for i,(events,target) in tqdm(enumerate(testset),total=len(testset)):
        new_events=change_speed_v2(events,params)
        test_data+=[{"events":new_events,"target":target}]

        # if i>50:
        #     break
    testset=ListNMNIST(test_data,transform=transform)
    test_sampler=RandomBatchSampler(testset,128)
    print(f"\033[92mdone\033[0m frame size : {len(testset[0][0])}")
    #<< テストデータの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> モデルの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device = torch.device(f"cuda:{args.device}")
    model_dirs=os.listdir(args.target)
    models={}
    for dir_name in model_dirs:

        if "beta" in dir_name.casefold() and "snn".casefold() in dir_name:
            config=load_yaml(str(Path(args.target)/dir_name/"conf.yml"))["model"]
            beta_snn=BetaCSNN(config,device=device).to(device)
            beta_snn.load_state_dict(torch.load(str(Path(args.target)/dir_name/"result/phase2_models/model_final.pth"),map_location=device))
            beta_snn.eval()
            models["beta-snn"]={"model":beta_snn}
            models["beta-snn"]["acc"]=[]

        elif "snn".casefold() in dir_name and not "beta".casefold() in dir_name and not "er".casefold() in dir_name:
            config=load_yaml(str(Path(args.target)/dir_name/"conf.yml"))["model"]["csnn"]
            csnn=CSNN(conf=config).to(device)
            csnn.load_state_dict(torch.load(str(Path(args.target)/dir_name/"result/models/model_final.pth"),map_location=device))
            csnn.eval()
            models["csnn"]={"model":csnn}
            models["csnn"]["acc"]=[]

        elif "lstm".casefold() in dir_name or "rnn".casefold() in dir_name: 
            config=load_yaml(str(Path(args.target)/dir_name/"conf.yml"))["model"]["crnn"]
            lstm=CLSTM(config).to(device)
            lstm.load_state_dict(torch.load(str(Path(args.target)/dir_name/"result/models/model_final.pth"),map_location=device))
            lstm.eval()
            models["lstm"]={"model":lstm}
            models["lstm"]["acc"]=[]

        elif "ersnn".casefold() in dir_name:
            config=load_yaml(str(Path(args.target)/dir_name/"conf.yml"))["model"]
            if "beta-cnn" in config.keys():
                beta_snn=ERSNN(config,device=device).to(device)
            elif "beta-lstm" in config.keys():
                beta_snn=ERSNNv2(config,device=device).to(device)
            beta_snn.load_state_dict(torch.load(str(Path(args.target)/dir_name/"result/phase2_models/model_final.pth")))
            beta_snn.eval()
            models["ersnn"]={"model":beta_snn}
            models["ersnn"]["acc"]=[]
    #<< モデルの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> テスト >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print("\nevaluating models...")
    with torch.no_grad():
        for i, (inputs, target) in tqdm(enumerate(test_sampler),total=len(test_sampler)):
            
            inputs[inputs==2]=1 #重なって2になってるとこは1にする
            inputs=torch.Tensor(inputs).to(device)
            inputs=torch.permute(inputs,dims=(1,0,2,3,4))
            target=torch.Tensor(target).to(device)
            # print("inputs:",inputs.shape,"targets:",target.shape)
            # print(torch.mean(inputs[0]),torch.mean(inputs[-1])) #0paddingの確認


            for key,item in models.items():
                if key=="beta-snn":
                    out,_=item["model"].forward(inputs,is_train_beta=True)
                    item["acc"]+=[SF.accuracy_rate(out,target)]
                elif key=="csnn":
                    out,_=item["model"](inputs)
                    item["acc"]+=[SF.accuracy_rate(out,target)]
                elif key=="lstm":
                    out=item["model"].forward(inputs)
                    out=torch.squeeze(out)
                    target=target.long()
                    item["acc"]+=[calculate_accuracy(out,target)]
                elif key=="ersnn":
                    out,_=item["model"].forward(inputs,is_beta=True,is_gamma=True)
                    item["acc"]+=[SF.accuracy_rate(out,target)]

    print("\033[92mdone\033[0m")
    #<< テスト <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # モデルのaccのmeanとstdを計算してprint
    print("\nresult"+"-"*30)
    for key, item in models.items():
        item["acc_mean"] = np.mean(item["acc"])
        item["acc_std"] = np.std(item["acc"])
        print(f"{key.upper()} Accuracy: Mean = {item['acc_mean']:.4f}, Std = {item['acc_std']:.4f}")


    # # 結果をCSVファイルに保存
    # save_results(savepath, beta_snn_acc_mean, beta_snn_acc_std, csnn_acc_mean, csnn_acc_std)

    import json
    # 結果をJSONファイルに保存
    results = {key: {"acc_mean": item["acc_mean"], "acc_std": item["acc_std"]} for key, item in models.items()}
    with open(savepath / "results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__=="__main__":
    main()