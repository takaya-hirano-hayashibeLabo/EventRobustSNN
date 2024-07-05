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
from scipy import stats

from utils import change_speed_v2,event2anim,get_random_subset,save_args,ListNMNIST,load_yaml,save_results,calculate_accuracy
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


def normalize(data, input_range=None, output_range=(0, 1)):
    if input_range is None:
        min_val = np.min(data)
        max_val = np.max(data)
    else:
        min_val, max_val = input_range

    out_min, out_max = output_range

    return ((data - min_val) / (max_val - min_val)) * (out_max - out_min) + out_min


def save_gamma_videos(gamma, inputs, save_dir, file_name, fr_trj, scale=9):
    """
    gamma: [timestep x channel x h x w] のサイズの動画データ
    inputs: [timestep x h_i x w_i] のサイズの入力データ
    save_dir: 保存先ディレクトリ
    file_name: ファイル名
    fr_trj: 描画する軌跡データ
    scale: 描画領域の拡大倍率
    """
    import subprocess
    import os
    import cv2
    import numpy as np

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # video shape: [timestep x channel x h x w]
    timestep, channel, h, w = gamma.shape
    assert h == w, "Height and width must be equal (square data)."
    
    # inputs shape: [timestep x h_i x w_i]
    _, h_i, w_i = inputs.shape
    assert h_i == w_i, "Height and width of inputs must be equal (square data)."
    
    # Apply scaling
    h_scaled = int(h * scale)
    left_width = int(2 * h_scaled / 3)
    color_bar_width = int(h_scaled / 6)
    total_width = int(left_width + h_scaled + color_bar_width)  # Total width is now 2/3*h_scaled + h_scaled + h_scaled/6
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    tmpout = str(save_dir / "tmp.avi")
    out = cv2.VideoWriter(tmpout, fourcc, 20.0, (total_width, h_scaled))
    
    grid_size = int(np.sqrt(channel))  # Assuming channel is a perfect square (e.g., 16 -> 4x4)
    cell_size = h_scaled // grid_size  # Size of each cell in the grid
    
    # Create a color bar
    color_bar = np.linspace(255, 0, h_scaled).astype(np.uint8)  # Reverse the color bar
    color_bar = np.tile(color_bar, (color_bar_width, 1)).T
    color_bar = cv2.applyColorMap(color_bar, cv2.COLORMAP_JET)
    
    for t in range(timestep):
        frame = np.zeros((h_scaled, total_width, 3), dtype=np.uint8)  # Frame size is now (h_scaled, total_width)
        
        # Left region (original)
        # Upper 2/3h region (inputs)
        upper_region = np.zeros((2*h_scaled//3, left_width, 3), dtype=np.uint8)
        input_frame = inputs[t, :, :].astype(np.float32)
        input_frame = cv2.normalize(input_frame, None, 0, 255, cv2.NORM_MINMAX)
        input_frame = cv2.applyColorMap(input_frame.astype(np.uint8), cv2.COLORMAP_JET)
        input_frame_resized = cv2.resize(input_frame, (left_width, 2*h_scaled//3))
        upper_region = input_frame_resized
        frame[:2*h_scaled//3, :left_width, :] = upper_region
        
        # Lower 1/3h region (only fr_trj)
        lower_region = np.zeros((h_scaled//3, left_width, 3), dtype=np.uint8)
        fr_t = [fr_trj[step] for step in range(t)]
        # gamma_mean=np.mean(gamma,axis=tuple(range(1,gamma.ndim)))
        # fr_t = [gamma_mean[step] for step in range(t)]
        max_fr_trj = max(fr_trj) if max(fr_trj) != 0 else 1  # Avoid division by zero
        for i in range(1, len(fr_t)):
            if fr_t[i - 1] is not None and fr_t[i] is not None:
                cv2.line(lower_region, 
                         (int((i - 1) * left_width / timestep), h_scaled//3 - int(fr_t[i - 1] / max_fr_trj * h_scaled//3)),
                         (int(i * left_width / timestep), h_scaled//3 - int(fr_t[i] / max_fr_trj * h_scaled//3)), 
                         (0, 255, 0), 1)  # Use bright color and thicker line
        frame[2*h_scaled//3:, :left_width, :] = lower_region
        
        # Middle region (new h x h region)

        # 全フレームの最大値と最小値を計算
        global_min = 0#np.min(gamma)
        global_max = 1#np.max(gamma)        
        for c in range(channel):
            heatmap = gamma[t, c, :, :]
            heatmap = normalize(heatmap,input_range=(global_min,global_max),output_range=(0,255))
            heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
            
            row = c // grid_size
            col = c % grid_size
            heatmap_resized = cv2.resize(heatmap, (cell_size, cell_size))
            frame[row*cell_size:(row+1)*cell_size, left_width+col*cell_size:left_width+(col+1)*cell_size, :] = heatmap_resized
        
        # Right region (new h x 1/6h region)
        color_bar = np.linspace(1, 0, h_scaled).astype(np.float32)  # Reverse the color bar
        color_bar = np.tile(color_bar, (color_bar_width, 1)).T
        color_bar = cv2.normalize(color_bar, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        color_bar = cv2.applyColorMap(color_bar.astype(np.uint8), cv2.COLORMAP_JET)
        frame[:, left_width + h_scaled:left_width + h_scaled + color_bar_width, :] = color_bar
        
        # Add min and max text to the color bar
        min_val, max_val = 0, 1  # Assuming the normalized range is [0, 1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White color for text
        thickness = 1

        # Position for min and max text
        min_pos = (left_width + h_scaled + color_bar_width, h_scaled - 5)
        max_pos = (left_width + h_scaled + color_bar_width, 15)

        cv2.putText(frame, f'{min_val:.2f}', min_pos, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'{max_val:.2f}', max_pos, font, font_scale, color, thickness, cv2.LINE_AA)
        
        
        # Write the frame
        out.write(frame)
    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    # ffmpegを使用して動画を再エンコード
    file_name = file_name + ".mp4" if not ".mp4" in file_name else file_name
    ffmpeg_command = [
        'ffmpeg', '-y', '-i', tmpout,
        '-pix_fmt', 'yuv420p', '-vcodec', 'libx264',
        '-crf', '23', '-preset', 'medium', str(save_dir / file_name)
    ]
    subprocess.run(ffmpeg_command)
    # 一時ファイルを削除
    os.remove(tmpout)



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
    datapath=Path(__file__).parent.parent.parent/"original_data"
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
            models["ersnn-v2"]["gamma-trj"]={}
            models["ersnn-v2"]["fr-trj"]={}
            models["ersnn-v2"]["inputs"]={}
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
                    _,gamma=item["model"].get_internal_params(inputs,is_beta=False)
                    # print("out shape:",out.shape,"target:", target)
                    for i,label in enumerate(target):
                        print(gamma[:,i].shape)
                        item["gamma-trj"][label.item()]=torch.squeeze(gamma[:,i]).to("cpu").numpy() #各ラベルのγ時系列
                        
                        #timestepごとの空間方向のfiring rateを記録する
                        fr_trj=[]
                        for sp_t in inputs[:,i]: 
                            fr_trj.append(torch.mean(sp_t).item())
                        item["fr-trj"][label.item()]=fr_trj

                        item["inputs"][label.item()]=(1.5*inputs[:,i,0]+0.5*inputs[:,i,1]-1).to("cpu").numpy()

    print("\033[92mdone\033[0m")
    #<< テスト <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    for data in testset:
        label=data[1]
        save_gamma_videos(
            gamma=models["ersnn-v2"]["gamma-trj"][label],
            inputs=models["ersnn-v2"]["inputs"][label],
            save_dir=savepath/"gamma-videos",
            file_name=f"label{label}",
            fr_trj=models["ersnn-v2"]["fr-trj"][label],
        )

        print(np.mean(models["ersnn-v2"]["gamma-trj"][label],axis=(1,2,3)))
if __name__=="__main__":
    main()