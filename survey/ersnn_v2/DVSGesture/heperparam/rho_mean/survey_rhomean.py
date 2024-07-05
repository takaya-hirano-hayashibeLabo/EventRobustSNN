"""
DVSGestureのデータサイズを調べる
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
import pandas as pd


def save_firing_trj_plot(firing_trj, filepath, filename):
    """
    firing_trjを描画して保存する関数
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(firing_trj)
    plt.title("Firing Trajectory")
    plt.xlabel("Time")
    plt.ylabel("Firing Rate")
    plt.savefig(filepath / f"{filename}.png")
    plt.close()

sensor_size=tonic.datasets.DVSGesture.sensor_size


time_window=50000
fps=int(1.0/(1e-6*time_window))

print(f"time window: {time_window} fps: {fps}"+"="*30)

size=64
scale=round(128/size)
transform=transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=time_window), #time_window msごとのデータをフレームデータに変換する
        torch.from_numpy,
        torchvision.transforms.Resize((size,size),antialias=True)
    ])


original_datapath=str("/mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202407_2MR/time_robust_snn_prj/original_data")
testset =  tonic.datasets.DVSGesture(save_to=original_datapath,  train=False,transform=transform)


for i in range(11):
    test_idx=i
    target=testset.targets[i]
    event_frame=testset[i][0]
    print(event_frame.shape)

    filepath=Path(__file__).parent/f"firing-trj"
    if not os.path.exists(filepath):
        os.makedirs(filepath)


    event_frame[event_frame>0]=1
    firing_trj=torch.mean(event_frame.to(float),dim=tuple(range(1,event_frame.ndim)))
    
    save_firing_trj_plot(
        firing_trj.detach().to("cpu").numpy(),
        filepath,f"label{target}"
    )