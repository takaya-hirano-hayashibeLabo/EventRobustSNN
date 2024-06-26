from copy import deepcopy
import numpy as np
from pathlib import Path
import json

from tonic.dataset import Dataset
from typing import Callable, Optional

def save_results(savepath, beta_snn_acc_mean, beta_snn_acc_std, csnn_acc_mean, csnn_acc_std):
    import csv
    results = [
        ["Model", "Mean Accuracy", "Standard Deviation"],
        ["Beta SNN", beta_snn_acc_mean, beta_snn_acc_std],
        ["CSNN", csnn_acc_mean, csnn_acc_std]
    ]
    
    with open(savepath / "results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)

def load_yaml(file_path):
    """
    Load a YAML file and return its contents as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML file contents.
    """
    import yaml
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def save_args(args, savepath):
    args_dict = vars(args)
    with open(savepath / "args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

def change_speed(events, alpha):
    """
    :param events: [(pixel_x, pixel_y, timestep, spike), ...]
    :param alpha: 速度倍率. 2にすれば2倍の速度になる
    """
    new_events=deepcopy(events)
    for i in range(len(events)):
        new_events[i][2] = deepcopy(int(new_events[i][2] / alpha))
    return new_events


def change_speed_trj(events,alpha_trj):
    """
    :param events: [timesteps x (pixel_x, pixel_y, timestep, spike)]
    :param alpha: 速度倍率 [timesteps]
    """
    dts=[]
    for i in range(1,len(events)):
        dts.append(int((events[i][2]-events[i-1][2])/alpha_trj[i]))
    new_events=deepcopy(events)
    for i in range(1,len(events)):
        new_events[i][2]=new_events[0][2]+sum(dts[:i])
    return new_events


def change_speed_v2(event, params:list):
    """
    指定した速度とフレーム比でイベントスピードを変換する

    :param event: [(pixel_x, pixel_y, timestep, spike), ...]
    :param params: [{speed, rate},...]   
                    ex) [{"speed":2.0, "rate":1.0},{"speed":0.5, "rate":1.0}]
    """


    #>> いい感じの比で速度変換の倍率リストを生成 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    split_rate=[]
    speeds=[]
    for p in params:
        speed=p["speed"]
        rate=p["rate"]
        speeds+=[speed]
        split_rate+=[speed*rate]

    event_num=len(event)
    speed_trj=np.zeros(event_num)
    idx=0
    for i, (s,r) in enumerate(zip(speeds,split_rate)):
        step=int(event_num*(r/np.sum(split_rate)))

        if i<len(speeds)-1:
            speed_trj[idx:idx+step]=s
            idx+=step
        else:
            speed_trj[idx:]=s
    #<< いい感じの比で速度変換の倍率リストを生成 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> 速度変換 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    dts=[]
    for i in range(1,len(event)):
        dts.append(int((event[i][2]-event[i-1][2])/speed_trj[i]))
    new_event=deepcopy(event)
    for i in range(1,len(event)):
        new_event[i][2]=new_event[0][2]+sum(dts[:i])
    #<< 速度変換 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    return new_event

    

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


def event2anim(events, time_window, max_x, max_y, output_path=Path(__file__).parent, file_name="output.mp4", scale_factor=10):
    """
    Convert event data to an animation and save as an mp4 file.

    Parameters:
    events (numpy structured array): The event data [(pixel_x, pixel_y, timestep, spike), ...]
    time_window (int): The time window for each frame.
    max_x (int): The maximum x value of the pixels.
    max_y (int): The maximum y value of the pixels.
    output_file (str): The path to the output mp4 file.
    scale_factor (int): Factor by which to scale the resolution.
    """
    import subprocess
    import os
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import viridis,plasma,cividis

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create a video writer object with increased resolution
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    scaled_max_x = max_x * scale_factor
    scaled_max_y = max_y * scale_factor
    tmpout = str(output_path / "tmp.avi")
    out = cv2.VideoWriter(tmpout, fourcc, 30.0, (scaled_max_x, scaled_max_y))

    # Determine the maximum timestep
    max_t = events[-1][2] + 1

    for frame_start in range(0, max_t, time_window):
        frame_end = frame_start + time_window

        frame = np.ones((scaled_max_y, scaled_max_x), dtype=np.uint8)*0.5

        # Add events to the frame
        for event in events:
            x, y, t, p = event
            if frame_start <= t < frame_end:
                scaled_x = x * scale_factor
                scaled_y = y * scale_factor
                color = 1 if p == 1 else 0  # Green for +1, Blue for -1
                for i in range(scale_factor):
                    for j in range(scale_factor):
                        frame[scaled_y + i, scaled_x + j] = color

        # Normalize the grayscale frame
        norm = Normalize(vmin=0, vmax=1)
        frame_normalized = norm(frame)

        # Apply the viridis colormap
        frame_colored = viridis(frame_normalized)

        # Convert the frame to uint8
        frame = (frame_colored * 255).astype(np.uint8)

        # Write the frame to the video
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Release the video writer object
    out.release()

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



def calculate_accuracy(output, target):
    """
    CNNやLSTMのACCを計算する関数
    """
    import torch
    predicted = torch.argmax(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy


class ListNMNIST(Dataset):
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
            save_to=str(Path(__file__).parent.parent.parent/"original_data"),
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