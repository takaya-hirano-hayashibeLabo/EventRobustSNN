import sys
from pathlib import Path
import shutil
import os

TERMINAL_WIDTH=shutil.get_terminal_size().columns

def get_subdirectories(path: str) -> list:
    """
    指定したパスの直下にあるディレクトリの名前をリストとして返す関数

    :param path: 対象のパス
    :return: ディレクトリ名のリスト
    """
    try:
        return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    except FileNotFoundError:
        print(f"指定したパスが見つかりません: {path}")
        return []
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return []


import h5py
from multiprocessing import Pool

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


def load_yaml(file_path):
    """
    Load a YAML file and return its contents as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML file contents.
    """
    import yaml
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

def get_minibatch(train_files_e, minibatch_size):
    import random
    random.shuffle(train_files_e)
    minibatch_files = train_files_e[:minibatch_size]
    train_files_e = train_files_e[minibatch_size:]
    return minibatch_files, train_files_e


def plot_results(csv_path, save_path):
    import pandas as pd
    import matplotlib.pyplot as plt

    # スタイルの設定
    plt.style.use('fivethirtyeight')
    # フォントの設定
    plt.rcParams['font.family'] = 'serif'
    # 背景色の設定
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    df = pd.read_csv(csv_path)
    
    epochs = df['ep']
    train_loss_mean = df['train_loss_mean']
    train_loss_std = df['train_loss_std']
    train_acc_mean = df['train_acc_mean']
    train_acc_std = df['train_acc_std']
    test_loss_mean = df['test_loss_mean']
    test_loss_std = df['test_loss_std']
    test_acc_mean = df['test_acc_mean']
    test_acc_std = df['test_acc_std']
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot training and test loss
    axs[0].plot(epochs, train_loss_mean, label='Train Loss', linestyle='-', marker='o')
    axs[0].fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2)
    axs[0].plot(epochs, test_loss_mean, label='Test Loss', linestyle='--', marker='x')
    axs[0].fill_between(epochs, test_loss_mean - test_loss_std, test_loss_mean + test_loss_std, alpha=0.2)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Test Loss')
    axs[0].legend()
    axs[0].grid(True, which='both', axis='both', linestyle='--', linewidth=2)
    
    # Plot training and test accuracy
    axs[1].plot(epochs, train_acc_mean, label='Train Accuracy', linestyle='-', marker='o')
    axs[1].fill_between(epochs, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std, alpha=0.2)
    axs[1].plot(epochs, test_acc_mean, label='Test Accuracy', linestyle='--', marker='x')
    axs[1].fill_between(epochs, test_acc_mean - test_acc_std, test_acc_mean + test_acc_std, alpha=0.2)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Training and Test Accuracy')
    axs[1].legend() 
    axs[1].grid(True, which='both', axis='both', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    for i in range(2):
        axs[i].spines['top'].set_color('white')
        axs[i].spines['right'].set_color('white')
        axs[i].spines['left'].set_color('white')
        axs[i].spines['bottom'].set_color('white')

    plt.savefig(save_path)
    plt.close()


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


def pad_sequences(data,time_sequence, device):
    import torch
    from torch.nn import functional as F
    import numpy as np

    # 最も長いtimesequenceの長さを取得
    max_timesequence = time_sequence
    
    # 各データを0パディングしてサイズを揃える
    padded_data = []
    for d in data:
        d=torch.Tensor(np.array(d)) #[timesequence x c x w x h]
        if not len(d)==max_timesequence:
            padding_size = (0, 0, 0, 0 ,0,0,0,max_timesequence - d.shape[0])
            padded_d = F.pad(d, padding_size, "constant", 0)
            padded_data.append(padded_d)
        else:
            padded_data.append(d)
    
    return torch.stack(padded_data).to(device)