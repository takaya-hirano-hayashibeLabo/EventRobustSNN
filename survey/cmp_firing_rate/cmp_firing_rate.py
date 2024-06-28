import h5py
from multiprocessing import Pool
import numpy as np

def load_single_hdf5(path):
    with h5py.File(path, 'r') as f:
        data = f['events'][:]
        target = f['target'][()]
    return data, target

def load_hdf5(file_path_list: list, num_workers: int = 64):
    """
    pathのリストからhdf5ファイルを読み込み, データを返す
    :return datas: [minibatch x frame]
    :return targets: [minibatch]
    """
    with Pool(num_workers) as pool:
        results = pool.map(load_single_hdf5, file_path_list)

    datas, targets = zip(*results)
    return list(datas), list(targets)

#>> 0.5倍速データ >>
test_x05_basedir="/mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202407_2MR/time_robust_snn_prj/framedata/NMNIST/speed_0.5_times/test/"
test_x05_files=[test_x05_basedir+f"data{i}.h5" for i in range(1)]
test_x05_datas,test_x05_targets=load_hdf5(test_x05_files)
x05_fr=np.mean(test_x05_datas,axis=(2,3,4)) #[batch x timestep] 空間方向に発火率をとる 
print(x05_fr.shape)

#>> 1倍速データ >>
test_x1_basedir="/mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202407_2MR/time_robust_snn_prj/framedata/NMNIST/speed_1.0_times/test/"
test_x1_files=[test_x1_basedir+f"data{i}.h5" for i in range(1)]
test_x1_datas,test_x1_targets=load_hdf5(test_x1_files)
x1_fr=np.mean(test_x1_datas,axis=(2,3,4)) #[batch x timestep] 空間方向に発火率をとる 
print(x1_fr.shape)

#>> 2倍速データ >>
test_x2_basedir="/mnt/ssd1/hiranotakaya/master/dev/workspace/prj_202407_2MR/time_robust_snn_prj/framedata/NMNIST/speed_2.0_times/test/"
test_x2_files=[test_x2_basedir+f"data{i}.h5" for i in range(1)]
test_x2_datas,test_x2_targets=load_hdf5(test_x2_files)
x2_fr=np.mean(test_x2_datas,axis=(2,3,4)) #[batch x timestep] 空間方向に発火率をとる 
print(x1_fr.shape)

alpha=1e3
r_2 =alpha*(x2_fr  -(x1_fr  ))**3
r_1 =alpha*(x1_fr  -(x1_fr  ))**3
r_05=alpha*(x05_fr-(x1_fr  ))**3

print("x1 fr:",np.mean(x1_fr,axis=-1))
print("x2 fr:",np.mean(x2_fr,axis=-1))
print("x05 fr:",np.mean(x05_fr,axis=-1))
print("-----")
print("x2/x1 fr:",np.mean(r_2,axis=1))
print("x05/x1 fr:",np.mean(r_05,axis=1))
# print(r)

# グラフの作成と保存
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

# フォントの設定
plt.rcParams['font.family'] = 'serif'
# 背景色の設定
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
cmap = matplotlib.colormaps.get_cmap('viridis')
# レイアウトの自動調整
plt.gca().spines['top'].set_color('white')
plt.gca().spines['right'].set_color('white')
plt.gca().spines['left'].set_color('white')
plt.gca().spines['bottom'].set_color('white')


plt.figure(figsize=(12, 8))  # Adjusted figure size to accommodate additional plots

plt.subplot(4, 1, 1)
plt.plot(x1_fr.T)
plt.title('Firing Rate for 1x Speed')
plt.xlabel('Timestep')
plt.ylabel('Firing Rate')

plt.subplot(4, 1, 2)
plt.plot(x2_fr.T)
plt.title('Firing Rate for 2x Speed')
plt.xlabel('Timestep')
plt.ylabel('Firing Rate')

plt.subplot(4, 1, 3)
plt.plot(x05_fr.T)
plt.title('Firing Rate for 0.5x Speed')
plt.xlabel('Timestep')
plt.ylabel('Firing Rate')

plt.subplot(4, 1, 4)
plt.plot(r_2.T, label='2x / 1x')
plt.plot(r_1.T, label='1x / 1x')
plt.plot(r_05.T, label='0.5x / 1x')
plt.title('Ratio of Firing Rates')
plt.xlabel('Timestep')
plt.ylabel('Ratio')
plt.legend()

plt.tight_layout()
plt.savefig(Path(__file__).parent/"firing_rates_comparison.png")