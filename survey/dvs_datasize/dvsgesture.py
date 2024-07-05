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

def save_heatmap_video(frames, output_path, file_name, fps=30, scale=5):
    import cv2
    import subprocess

    height, width = frames[0].shape
    new_height, new_width = int(height * scale), int(width * scale)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    tmpout = str(output_path / "tmp.avi")
    video = cv2.VideoWriter(tmpout, fourcc, fps, (new_width, new_height), isColor=True)

    for frame in frames:
        # Normalize frame to range [0, 255] with original range [-1, 1]
        normalized_frame = ((frame + 1) / 2 * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_JET)
        resized_heatmap = cv2.resize(heatmap, (new_width, new_height))
        video.write(resized_heatmap)

    video.release()

    # Re-encode the video using ffmpeg
    file_name = file_name + ".mp4" if not ".mp4" in file_name else file_name
    ffmpeg_command = [
        'ffmpeg', '-y', '-i', tmpout,
        '-pix_fmt', 'yuv420p', '-vcodec', 'libx264',
        '-crf', '23', '-preset', 'medium', str(output_path / file_name)
    ]
    subprocess.run(ffmpeg_command)
    # Remove the temporary file
    os.remove(tmpout)


sensor_size=tonic.datasets.DVSGesture.sensor_size



datasize=[]
for time_window in range(10000, 100000, 10000):

    fps=int(1.0/(1e-6*time_window))

    print(f"time window: {time_window} fps: {fps}"+"="*30)

    size=32
    scale=round(128/size)
    transform=transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=time_window), #time_window msごとのデータをフレームデータに変換する
            torch.from_numpy,
            torchvision.transforms.Resize((size,size),antialias=True)
            # torchvision.transforms.Resize((64, 64),antialias=True)
        ])


    original_datapath=str("/mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202407_2MR/time_robust_snn_prj/original_data")
    trainset = tonic.datasets.DVSGesture(save_to=original_datapath,  train=True,transform=transform)
    testset =  tonic.datasets.DVSGesture(save_to=original_datapath,  train=False,transform=transform)

    print("Sensor size: ",sensor_size)
    print("Trainset size:", len(trainset),"Testset size:", len(testset))
    print("Sequence Length: ",trainset[0][0].shape) #1つのイベントデータの時系列長さ    
    # testsetのクラス数を表示
    num_classes = len(set(testset.targets)) #11コおきに順番にテストデータが並んでいる
    print("Number of classes:", num_classes)


    for i in range(11):
        test_idx=i
        target=testset.targets[i]
        event_frame=testset[i][0]
        print(event_frame.shape)

        filepath=Path(__file__).parent/f"window_{time_window}"
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        
        frames=1.5*event_frame[:,0]+0.5*event_frame[:,1]-1 #[timestep x h x w]

        save_heatmap_video(
            frames.detach().to("cpu").numpy(),filepath,f"target{target}",
            scale=scale,fps=fps
        )

        # if i>0:
        #     break

    eventset=tonic.datasets.DVSGesture(save_to=original_datapath,  train=False)
    print(f"event sequence: {eventset[0][0].shape}, event shape: {eventset[0][0][0]}")
    # exit(1)



    datasize.append([time_window,trainset[0][0].shape[0]])

    # break

datasize=pd.DataFrame(datasize,columns=["time windows","sequence size"])
datasize.to_csv(Path(__file__).parent/"datasize.csv",index=False)