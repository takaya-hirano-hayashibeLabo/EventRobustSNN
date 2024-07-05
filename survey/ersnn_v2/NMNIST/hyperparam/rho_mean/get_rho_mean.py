# 1倍速のときの平均イベント密度を調査

from pathlib import Path
import sys
ROOT=Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(ROOT)) #root
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
from scipy import stats

from experiment.utils import change_speed_v2,event2anim,get_random_subset,save_args,ListNMNIST,load_yaml,save_results,calculate_accuracy
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



def timestep2frame_speed(new_events,ts_speed_trj,window_size=1400):
    """
    タイムスタンプでの速度倍率リストをフレームでの速度倍率リストに変換する関数
    """
    # print(f"new_events: {len(new_events)}, speed-trj: {len(ts_speed_trj)}")
    max_t = new_events[-1][2] + 1
    min_t=new_events[0][2]

    frame_speed_trj=[]
    prev_mode=1.0 #前回の最頻値
    for frame_start in range(min_t, max_t, window_size):
        frame_end = frame_start + window_size if frame_start + window_size<=max_t else max_t

        s_trj=[]
        for idx,e in enumerate(new_events):
            x, y, t, p = e
            if frame_start <= t < frame_end:
                s_trj.append(ts_speed_trj[idx])
        
        # print(f"start: {frame_start}, end: {frame_end}, t: {t}",s_trj)
        mode=stats.mode(s_trj).mode
        if len(s_trj)==0:
            mode=prev_mode
        frame_speed_trj.append(mode)

        prev_mode=mode

    return frame_speed_trj



def plot_fr_trj(fr,save_dir:Path,file_name):
    import matplotlib.pyplot as plt
    # スタイルの設定
    plt.style.use('fivethirtyeight')

    # フォントの設定
    plt.rcParams['font.family'] = 'serif'
    # 背景色の設定
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    fig=plt.figure()
    plt.plot(fr)
    fr_mean=np.mean(fr)
    fr_mean_plt=np.ones_like(fr)*fr_mean
    plt.plot(fr_mean_plt,ls="--")
    # 平均値をテキストで描画
    plt.text(0.5, 0.5, f"Mean: {fr_mean}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='center', horizontalalignment='center')
    plt.title(f"{file_name} fr trj")

    # レイアウトの自動調整
    plt.tight_layout()
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['bottom'].set_color('white')

    plt.savefig(str(save_dir/file_name)+".png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str,help="各モデルが入ったディレクトリ",required=True)
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
    datapath=ROOT/"original_data"
    testset = tonic.datasets.NMNIST(save_to=str(datapath), train=False)
    # for i,t in enumerate(testset):
    #     print(f"idx[{i}] label:",t[1])
    testset=[testset[0],testset[5000],testset[9999]] #検証するテストデータ3つ


    sensor_size = tonic.datasets.NMNIST.sensor_size
    transform=transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=1400), #time_window msごとのデータをフレームデータに変換する
            torch.from_numpy,
            torchvision.transforms.Resize((32, 32),antialias=True)
        ])


    params=[]
    for s,r in zip(args.speed,args.rate):
        params+=[
            {"speed":s, "rate":r}
        ]
    print("\nchanging test event speed...")
    test_data=[]
    speed_trjs={}
    for i,(events,target) in tqdm(enumerate(testset),total=len(testset)):
        new_events,speed_trj=change_speed_v2(events,params,is_return_speed_trj=True)
        test_data+=[{"events":new_events,"target":target}]
        speed_trjs[target]=timestep2frame_speed(new_events,speed_trj,1400)
        # event2anim(new_events,1400,34,34,   output_path=savepath/"videos",file_name=f"event{target}_changed")

    testset=ListNMNIST(test_data,transform=transform)
    test_sampler=RandomBatchSampler(testset,batch_size=1)

    print("label[",testset[0][1],"]",len(testset[0][0]),len(speed_trjs[testset[0][1]]))
    #<< テストデータの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> モデルの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    device = torch.device(f"cuda:{args.device}")
    model_dirs=os.listdir(args.target)
    models={}
    for dir_name in model_dirs:

        if "ersnn".casefold() in dir_name and "v2".casefold() in dir_name:
            config=load_yaml(str(Path(args.target)/dir_name/"conf.yml"))["model"]
            beta_snn=ERSNNv2(config,device=device).to(device)
            beta_snn.load_state_dict(torch.load(str(Path(args.target)/dir_name/"result/phase2_models/model_final.pth")))
            beta_snn.eval()
            models["ersnn-v2"]={"model":beta_snn}
            models["ersnn-v2"]["fr-trj"]={}
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
                if key=="ersnn-v2":
                    for i,label in enumerate(target):
                        
                        #timestepごとの空間方向のfiring rateを記録する
                        fr_trj=[]
                        for sp_t in inputs[:,i]: 
                            fr_trj.append(torch.mean(sp_t).item())
                        item["fr-trj"][label.item()]=fr_trj


    print("\033[92mdone\033[0m")
    #<< テスト <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    for data in testset:
        label=data[1]
        plot_fr_trj(
            fr=models["ersnn-v2"]["fr-trj"][label],
            save_dir=savepath,
            file_name=f"label{label}",
        )
if __name__=="__main__":
    main()