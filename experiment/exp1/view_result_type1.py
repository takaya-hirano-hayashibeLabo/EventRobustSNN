"""
速度を全体的に変えたときのスコア変化
"""

import os
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def main():
    parser = argparse.ArgumentParser(description="指定したパスの1つ下のディレクトリを取得")
    parser.add_argument('--target', type=str, help='対象のパス')
    args = parser.parse_args()

    # 指定したパスの1つ下のディレクトリを取得
    target_dir=Path(args.target)
    dirs=os.listdir(target_dir)
    # print(dirs)

    data_table = []
    for dir_name in dirs:
        json_path = target_dir / dir_name / 'results.json'
        args_path = target_dir / dir_name / 'args.json'
        if json_path.exists() and args_path.exists():
            with open(json_path, 'r') as f:
                results_data = json.load(f)
            with open(args_path, 'r') as f:
                args_data = json.load(f)
            data_entry = {'speed': args_data['speed'][0]}  # speedの最初の値を取得
            for key, value in results_data.items():
                data_entry[f'{key}_acc_mean'] = value['acc_mean']
                data_entry[f'{key}_acc_std'] = value['acc_std']
            data_table.append(data_entry)


    # pandasのDataFrameに変換
    df = pd.DataFrame(data_table)
    df=df.sort_values(by='speed')
    print(df)

    # speedが1.0の行を取得
    base_row = df[df['speed'] == 1.0].iloc[0]
    df_normalized = df.copy()
    for key in results_data.keys():
        base_acc_mean = base_row[f'{key}_acc_mean']
        base_acc_std = base_row[f'{key}_acc_std']
        df_normalized[f'{key}_acc_mean'] = df[f'{key}_acc_mean'] / base_acc_mean
        df_normalized[f'{key}_acc_std'] = df_normalized[f'{key}_acc_mean'] * ((df[f'{key}_acc_std'] / df[f'{key}_acc_mean'])**2 + (base_acc_std / base_acc_mean)**2)**0.5

    print("Normalized DataFrame:")
    print(df_normalized)


    # グラフを描画

    # スタイルの設定
    plt.style.use('fivethirtyeight')

    # フォントの設定
    plt.rcParams['font.family'] = 'serif'
    # 背景色の設定
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.figure(figsize=(10, 6))
    cmap = matplotlib.colormaps.get_cmap('viridis')

    for i, key in enumerate(results_data.keys()):
        plt.plot(df_normalized['speed'], df_normalized[f'{key}_acc_mean'], label=key.upper(), color=cmap(i / len(results_data)), linestyle='-')
        plt.fill_between(df_normalized['speed'], 
                         df_normalized[f'{key}_acc_mean'] - df_normalized[f'{key}_acc_std'], 
                         df_normalized[f'{key}_acc_mean'] + df_normalized[f'{key}_acc_std'], 
                         color=cmap(i / len(results_data)), alpha=0.2)

    plt.xlabel('Speed')
    plt.ylabel('Normalized Accuracy')
    plt.ylim(0, 1.1)  # y軸の範囲を0から1に設定
    plt.title('Normalized Accuracy vs Speed')
    plt.legend()
    plt.grid(True)

    # レイアウトの自動調整
    plt.tight_layout()
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['bottom'].set_color('white')


    plt.savefig(target_dir/"result.png")

if __name__=="__main__":
    main()