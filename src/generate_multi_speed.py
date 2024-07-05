"""
指定したspeedのイベントデータを生成して保存する
"""
import os
from typing import Callable, Optional
import tonic.transforms as transforms
import tonic
from tonic.dataset import Dataset
from pathlib import Path
import numpy as np
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import torch
import torchvision
import shutil
import h5py
from copy import deepcopy

TERMINAL_WIDTH=shutil.get_terminal_size().columns

def change_speed(events, alpha):
    """
    :param events: [(pixel_x, pixel_y, timestep, spike), ...]
    :param alpha: 速度倍率. 2にすれば2倍の速度になる
    """
    new_events=deepcopy(events)
    for i in range(len(events)):
        new_events[i][2] = deepcopy(int(new_events[i][2] / alpha))
    return new_events

def create_windows(data, window_size, overlap):
    """
    長いシーケンスデータをオーバーラップを持たせてwindowごとに区切る
    """
    step = window_size - overlap
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        if i + window_size <= len(data):
            windows.append(data[i:i + window_size])
    return windows


def cut_string_end(s, size):
    """
    文字列の末尾を指定したサイズでカットする

    Parameters:
    s (str): 元の文字列
    size (int): 出力する文字列のサイズ

    Returns:
    str: 指定されたサイズの末尾の文字列
    """
    if size < 0:
        raise ValueError("Size must be non-negative")
    return s[:size] if size <= len(s) else s


def save_to_hdf5(file_path, data, target):
    """
    hdf5で圧縮して保存
    そうしないとTオーダのデータサイズになってしまう
    """
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('events', data=data.numpy().astype(np.int8) , compression='gzip')
        f.create_dataset('target', data=target)


def get_random_subset(dataset, fraction):
    """
    指定した割合のデータをランダムに取り出す関数

    Parameters:
    dataset (Dataset): 元のデータセット
    fraction (float): 取り出すデータの割合 (0 < fraction <= 1)

    Returns:
    Subset: ランダムに取り出されたデータのサブセット
    """
    from torch.utils.data import Subset
    import random

    dataset_size = len(dataset)
    subset_size = int(dataset_size * fraction)
    indices = random.sample(range(dataset_size), subset_size)
    return Subset(dataset, indices)


class NpyNMNISTFromPath(Dataset):
    """
    Custom Dataset for loading data from .npy files.

    Parameters:
        data_path (string): Location of the .npy files.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """

    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            data_path,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.data_path = data_path
        self.data = []
        self.targets = []

        # Load .npy files
        print(f"loading {data_path}...")
        for file in tqdm(os.listdir(data_path)):
            if file.endswith(".npy"):
                file_path = os.path.join(data_path, file)
                data = np.load(file_path, allow_pickle=True).item()
                self.data.append(data['events'])
                self.targets.append(data['target'])
        print("\033[92mdone\033[0m")

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target
            class.
        """
        events = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target

    def __len__(self) -> int:
        return len(self.data)
    
class ListDVSDataset(Dataset):
    """
    Listデータからdataset作るクラス

    Parameters:
        data_path (string): Location of the .npy files.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """

    def __init__(
        self,
        data_list: list,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            save_to=str(Path(__file__).parent.parent/"original_data"),
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.data = []
        self.targets = []

        # Load .npy files
        # print(f"creating dataset...")
        for item in (data_list):
            self.data.append(item['events'])
            self.targets.append(item['target'])
        # print("\033[92mdone\033[0m")

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target
            class.
        """
        events = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target

    def __len__(self) -> int:
        return len(self.data)


def generate_multi_dataset():

    sensor_size=(34,34,2)

    transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=1000)
        ])

    original_datapath=str(Path(__file__).parent.parent/"original_data")
    trainset = tonic.datasets.NMNIST(save_to=original_datapath, transform=transform, train=True)
    testset = tonic.datasets.NMNIST(save_to=original_datapath, transform=transform, train=False)

    savepath=Path(__file__).parent.parent / "data"

    alpha_list=[0.5,1.0,2.0] #この速度のデータを保存する
    for a in alpha_list:
        print(f"saving {a} times speed datas... ")
        print("[trainset]")
        savepath_a=savepath/f"speed_{a}_times/train"
        if not os.path.exists(savepath_a):
            os.makedirs(savepath_a)

        for i, (events, target) in tqdm(enumerate(trainset),total=len(trainset)):
            np.save(savepath_a / f"data{i}.npy", {'events': events, 'target': target})

        print("[testset]")
        savepath_a=savepath/f"speed_{a}_times/test"
        if not os.path.exists(savepath_a):
            os.makedirs(savepath_a)

        for i, (events, target) in tqdm(enumerate(testset),total=len(testset)):
            np.save(savepath_a / f"data{i}.npy", {'events': events, 'target': target})
        print("\033[92mdone\033[0m")


def generate_multi_dataframe(datatype:str,data_rate:float,resize_to=(32,32), time_window=1400, speed_list=[1.0], sequence_size=100):
    """
    frameデータとして保存する関数  
    学習時はミニバッチごとにディレクトリから集める  
    そうしないと, データがでかすぎてメモリが死ぬ

    :param <str>datatype: {NMNIST, DVSGesture}
    :param data_rate: 何割にデータを間引くか
    :param <tuple>resize_to: resize後のデータの(height x width)
    :param time_window: 1フレームあたり何time_window含むか. 単位はイベントのタイムスタンプによる
    :param <list>speed_list: 生成するデータの速度倍率のリスト
    :param sequence_size: フレーム変換後のデータのシーケンス長さ. この長さ分学習に時系列として入力する
    """

    original_datapath=str(Path(__file__).parent.parent/"original_data")
    if datatype.casefold()=="NMNIST".casefold():
        sensor_size = tonic.datasets.NMNIST.sensor_size
        trainset = tonic.datasets.NMNIST(save_to=original_datapath, train=True)
        testset = tonic.datasets.NMNIST(save_to=original_datapath,  train=False)

    elif datatype.casefold()=="DVSGesture".casefold():
        sensor_size=tonic.datasets.DVSGesture.sensor_size
        trainset = tonic.datasets.DVSGesture(save_to=original_datapath, train=True)
        testset =  tonic.datasets.DVSGesture(save_to=original_datapath,  train=False)

    else:
        print(f"unkown datatype '{datatype}'...")
        exit(1)

    print(len(trainset))
    print(trainset[0][0].shape)

    trainset=get_random_subset(trainset,data_rate) #データを間引く
    testset=get_random_subset(testset,data_rate) #データを間引く

    # exit(1)

    transform=transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=time_window), #time_window msごとのデータをフレームデータに変換する
            torch.from_numpy,
            torchvision.transforms.Resize(resize_to,antialias=True)
        ])
    # cache_transform = tonic.transforms.Compose([torch.from_numpy,
    #                                     torchvision.transforms.Resize((32, 32)),])

    savepath=Path(__file__).parent.parent / "framedata"

    alpha_list=speed_list #この速度のデータを保存する
    window_size=sequence_size 
    for i,a in enumerate(alpha_list):
        print("\n\033[96m"+cut_string_end(
            f"PROCESS [{i+1}/{len(alpha_list)}] @ALPHA={a}"+"="*TERMINAL_WIDTH,size=TERMINAL_WIDTH)+"\033[0m"
        )

        #>> tarin dataの生成 >>
        print(cut_string_end("[trainset]"+"-"*TERMINAL_WIDTH,size=TERMINAL_WIDTH))
        savepath_a=savepath/f"speed_{a}_times/train"
        if not os.path.exists(savepath_a):
            os.makedirs(savepath_a)

        print(f"convert to {a} times speed... ")
        train_data=[]
        for i, (events, target) in tqdm(enumerate(trainset),total=len(trainset)):
            events=change_speed(events,a)
            train_data+=[{"events":events,"target":target}]
        print("\033[92mdone\033[0m")

        print("saving frame datas...")
        trainframe=ListDVSDataset(train_data,transform=transform) #フレームデータに変換
        count=0
        min_steps=1e10
        for (data, target) in tqdm((trainframe),total=len(trainframe)):
            # print(data[data==1].shape,data[data==2].shape)
            data[data==2]=1 #倍速にするほど2の位置が増えるので1に強制変換. 2のピクセル数は1のピクセル数に比べて少ないので, 変換しても大丈夫と思われる
            # print(data[data==1].shape,data[data==2].shape)
            if data.shape[0]>window_size:
                window_data=create_windows(data,window_size,overlap=int(window_size/2))
                for d in window_data:
                    save_to_hdf5(savepath_a / f"data{count}.h5",d,target)
                    count+=1
            else:
                save_to_hdf5(savepath_a / f"data{count}.h5",data,target)
                count+=1

            if min_steps>data.shape[0]:
                # print("min data shape: ",data.shape)
                min_steps=data.shape[0]
        print("\033[92mdone\033[0m")
        # print(f"data shape: {data.shape}")
        # exit(1)
        #>> tarin dataの生成 >>

        #>> test dataの生成 >>
        print(cut_string_end("[testset]"+"-"*TERMINAL_WIDTH,size=TERMINAL_WIDTH))
        savepath_a=savepath/f"speed_{a}_times/test"
        if not os.path.exists(savepath_a):
            os.makedirs(savepath_a)

        print(f"convert to {a} times speed... ")
        test_data=[]
        for i, (events, target) in tqdm(enumerate(testset),total=len(testset)):
            events=change_speed(events,a)
            test_data+=[{"events":events,"target":target}]
        print("\033[92mdone\033[0m")

        print("saving frame datas...")
        testframe=ListDVSDataset(test_data,transform=transform) #フレームデータに変換
        count=0
        min_steps=1e10
        for (data, target) in tqdm((testframe),total=len(testframe)):
            if data.shape[0]>window_size:
                window_data=create_windows(data,window_size,overlap=int(window_size/2))
                for d in window_data:
                    # np.save(savepath_a / f"data{count}.npy", {'events': d, 'target': target},allow_pickle=True)
                    save_to_hdf5(savepath_a / f"data{count}.h5",d,target)
                    count+=1

            else:
                # np.save(savepath_a / f"data{count}.npy", {'events': data, 'target': target},allow_pickle=True)
                save_to_hdf5(savepath_a / f"data{count}.h5",data,target)
                count+=1

            if min_steps>data.shape[0]:
                # print("min data shape: ",data.shape)
                min_steps=data.shape[0]
        print("\033[92mdone\033[0m")
        #>> test dataの生成 >>

def load_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['events'][:]
        target = f['target'][()]
    return {'events': data, 'target': target}

def test():
    """
    データの保存からspeed変えたのをnpyに保存し, さらに学習できるレベルに変換するワークフロー
    """
    dataset = tonic.datasets.NMNIST(save_to=str(Path(__file__).parent.parent/"data"), train=True)
    sensor_size = tonic.datasets.NMNIST.sensor_size
    print(len(dataset),sensor_size)

    save_path = Path(__file__).parent.parent / "npy_data"
    save_path.mkdir(parents=True, exist_ok=True)

    max_count=100
    for i, (events, target) in enumerate(dataset):
        np.save(save_path / f"sample_{i}.npy", {'events': events, 'target': target})
        if i>max_count:
            break
    print(f"Dataset saved to {save_path}")


    #>> 一旦, npy形式から読み込む >>
    #このフェーズで, いろんなspeedのデータを1つのdatasetに結合できればいい
    transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=1000)
        ])
    dataset_np1=NpyNMNISTFromPath(
        data_path=save_path,transform=transform
    )
    dataset_np2=NpyNMNISTFromPath(
        data_path=save_path,transform=transform
    )
    dataset_np = ConcatDataset([dataset_np1, dataset_np2]) #これで結合できる
    #<< 一旦, npy形式から読み込む <<


    #>> 読み込んだdatasetをキャッシュ上でのdatasetに変換する >>
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from tonic import DiskCachedDataset

    transform = tonic.transforms.Compose([torch.from_numpy,
                                        torchvision.transforms.RandomRotation([-10,10])])

    cached_trainset = DiskCachedDataset(dataset_np, transform=transform, cache_path='./cache/nmnist/train')

    batch_size = 128
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
    #<< 読み込んだdatasetをキャッシュ上でのdatasetに変換する <<
    

    print("dataset_np shape: ",len(dataset_np))

    frame,target=next(iter(trainloader))
    print("frame:\n",frame)
    print(f"target: {target}")


def main():
    import argparse
    import json

    parser=argparse.ArgumentParser()
    parser.add_argument("--datatype",default="NMNIST")
    parser.add_argument("--data_rate",default=0.1,type=float)
    parser.add_argument("--resize_to", nargs="+",type=int,required=True,help="変換後のフレームサイズ")
    parser.add_argument("--time_window",type=int,default=1400,help="eventをtoFrameするときのtime_window")
    parser.add_argument("--speed_list",nargs="+",type=float,required=True,help="生成する速度倍率リスト")
    parser.add_argument("--sequence_size",default=100,type=int,help="最終的に生成する学習用フレームデータのシーケンス長")
    args=parser.parse_args()


    # Save args to a JSON file
    args_dict = vars(args)
    savepath=Path(__file__).parent.parent / "framedata"
    with open(savepath/'args.json', 'w') as f:
        json.dump(args_dict, f, indent=4)


    generate_multi_dataframe(
        args.datatype,args.data_rate,
        resize_to=tuple(args.resize_to),
        time_window=args.time_window,
        speed_list=args.speed_list,
        sequence_size=args.sequence_size
    )


if __name__=="__main__":
    # test()
    # generate_multi_dataset()
    main()

    # data=load_from_hdf5("/mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202407_2MR/time_robust_snn_prj/framedata/speed_0.5_times/test/data0.h5")
    # print(data)
    # print(np.max(data["events"]),np.min(data["events"]))