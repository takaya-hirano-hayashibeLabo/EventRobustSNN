"""
時系列変化に対する変化の描画
"""

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
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="指定したパスの1つ下のディレクトリを取得")
    parser.add_argument('--target', type=str, help='対象のパス')
    parser.add_argument("--base_dir",type=str,help="1倍のときのresult")
    args = parser.parse_args()

    # 指定したパスの1つ下のディレクトリを取得
    target_dir=Path(args.target)
    dirs=os.listdir(target_dir)
    # print(dirs)

    # data_table = []
    # for dir_name in dirs:
    #     json_path = target_dir / dir_name / 'results.json'
    #     args_path = target_dir / dir_name / 'args.json'
    #     if json_path.exists() and args_path.exists():
    #         with open(json_path, 'r') as f:
    #             results_data = json.load(f)
    #         with open(args_path, 'r') as f:
    #             args_data = json.load(f)
    #         data_table.append({
    #             # 'dir_name': dir_name,
    #             'speed': "-".join([str(s) for s in args_data['speed']]),  # speed
    #             'lstm_acc_mean': results_data['lstm']['acc_mean'],
    #             'lstm_acc_std': results_data['lstm']['acc_std'],
    #             'csnn_acc_mean': results_data['csnn']['acc_mean'],
    #             'csnn_acc_std': results_data['csnn']['acc_std'],
    #             'beta_snn_acc_mean': results_data['beta-snn']['acc_mean'],
    #             'beta_snn_acc_std': results_data['beta-snn']['acc_std']
    #         })

    data_table = []
    for dir_name in dirs:
        json_path = target_dir / dir_name / 'results.json'
        args_path = target_dir / dir_name / 'args.json'
        if json_path.exists() and args_path.exists():
            with open(json_path, 'r') as f:
                results_data = json.load(f)
            with open(args_path, 'r') as f:
                args_data = json.load(f)
            data_entry = {'speed': "-".join([str(s) for s in args_data['speed']])}  # speedの最初の値を取得
            for key, value in results_data.items():
                data_entry[f'{key}_acc_mean'] = value['acc_mean']
                data_entry[f'{key}_acc_std'] = value['acc_std']
            data_table.append(data_entry)
    
    base_dir=Path(args.base_dir)
    base_dir_json_path = base_dir / 'results.json'
    base_dir_args_path = base_dir / 'args.json'
    with open(base_dir_json_path, 'r') as f:
        base_results_data = json.load(f)
    with open(base_dir_args_path, 'r') as f:
        base_args_data = json.load(f)
    data_entry = {'speed': str(base_args_data['speed'][0])}  # speedの最初の値を取得
    for key, value in base_results_data.items():
        data_entry[f'{key}_acc_mean'] = value['acc_mean']
        data_entry[f'{key}_acc_std'] = value['acc_std']
    data_table.append(data_entry)

    # pandasのDataFrameに変換
    df = pd.DataFrame(data_table)
    df=df.sort_values(by='speed')
    print(df)

    # speedが1.0の行を取得
    if not df[df['speed'] == "1.0"].empty:
        base_row = df[df['speed'] == "1.0"].iloc[0]
        df_normalized = df.copy()
        for key in results_data.keys():
            base_acc_mean = base_row[f'{key}_acc_mean']
            base_acc_std = base_row[f'{key}_acc_std']
            df_normalized[f'{key}_acc_mean'] = df[f'{key}_acc_mean'] / base_acc_mean
            df_normalized[f'{key}_acc_std'] = df_normalized[f'{key}_acc_mean'] * ((df[f'{key}_acc_std'] / df[f'{key}_acc_mean'])**2 + (base_acc_std / base_acc_mean)**2)**0.5
    else:
        print("Error: speedが1.0の行が存在しません。")
        return
    
    print("Normalized DataFrame:")
    print(df_normalized)


    # 棒グラフの描画
    # スタイルの設定
    plt.style.use('fivethirtyeight')

    # フォントの設定
    plt.rcParams['font.family'] = 'serif'
    # 背景色の設定
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    cmap = matplotlib.colormaps.get_cmap('viridis')
    
    speeds = df_normalized['speed'].values
    # lstm_acc_means = df_normalized['lstm_acc_mean'].values
    # lstm_acc_stds = df_normalized['lstm_acc_std'].values
    # csnn_acc_means = df_normalized['csnn_acc_mean'].values
    # csnn_acc_stds = df_normalized['csnn_acc_std'].values
    # beta_snn_acc_means = df_normalized['beta_snn_acc_mean'].values
    # beta_snn_acc_stds = df_normalized['beta_snn_acc_std'].values

    x = np.arange(len(speeds))  # 速度のインデックス
    width = 0.2  # 各棒の幅

    fig, ax = plt.subplots(figsize=(12, 6))

    # 各モデルの棒グラフを描画
    for i, key in enumerate(results_data.keys()):
        means = df_normalized[f'{key}_acc_mean'].values
        stds = df_normalized[f'{key}_acc_std'].values
        ax.bar(x + i * width, means, width, label=key.upper(), yerr=stds, capsize=5, color=cmap(i / len(results_data)))

    # ラベルとタイトルの設定
    ax.set_xlabel('Speed')
    ax.set_ylabel('Accuracy (Normalized)')
    ax.set_title('Normalized Accuracy by Speed')
    ax.set_xticks(x + width * (len(results_data) - 1) / 2)
    ax.set_xticklabels(df_normalized['speed'])
    ax.legend()

    # グラフの表示
    plt.ylim(0, 1.1)  # 割合は1を超えないように設定

    # レイアウトの自動調整
    plt.tight_layout()
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['bottom'].set_color('white')

    plt.savefig(target_dir/"result.png")


if __name__=="__main__":
    main()