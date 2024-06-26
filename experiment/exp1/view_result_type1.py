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
            data_table.append({
                # 'dir_name': dir_name,
                'speed': args_data['speed'][0],  # speedの最初の値を取得
                'lstm_acc_mean': results_data['lstm']['acc_mean'],
                'lstm_acc_std': results_data['lstm']['acc_std'],
                'csnn_acc_mean': results_data['csnn']['acc_mean'],
                'csnn_acc_std': results_data['csnn']['acc_std'],
                'beta_snn_acc_mean': results_data['beta-snn']['acc_mean'],
                'beta_snn_acc_std': results_data['beta-snn']['acc_std']
            })
    
    # pandasのDataFrameに変換
    df = pd.DataFrame(data_table)
    df=df.sort_values(by='speed')
    print(df)

    # speedが1.0の行を取得
    base_row = df[df['speed'] == 1.0].iloc[0]
    base_lstm_acc_mean = base_row['lstm_acc_mean']
    base_lstm_acc_std = base_row['lstm_acc_std']
    base_csnn_acc_mean = base_row['csnn_acc_mean']
    base_csnn_acc_std = base_row['csnn_acc_std']
    base_beta_snn_acc_mean = base_row['beta_snn_acc_mean']
    base_beta_snn_acc_std = base_row['beta_snn_acc_std']

    # 新しいテーブルを作成
    df_normalized = df.copy()
    df_normalized['lstm_acc_mean'] = df['lstm_acc_mean'] / base_lstm_acc_mean
    df_normalized['csnn_acc_mean'] = df['csnn_acc_mean'] / base_csnn_acc_mean
    df_normalized['beta_snn_acc_mean'] = df['beta_snn_acc_mean'] / base_beta_snn_acc_mean

    # 誤差伝播の法則を利用して標準偏差を計算
    df_normalized['lstm_acc_std'] = df_normalized['lstm_acc_mean'] * ((df['lstm_acc_std'] / df['lstm_acc_mean'])**2 + (base_lstm_acc_std / base_lstm_acc_mean)**2)**0.5
    df_normalized['csnn_acc_std'] = df_normalized['csnn_acc_mean'] * ((df['csnn_acc_std'] / df['csnn_acc_mean'])**2 + (base_csnn_acc_std / base_csnn_acc_mean)**2)**0.5
    df_normalized['beta_snn_acc_std'] = df_normalized['beta_snn_acc_mean'] * ((df['beta_snn_acc_std'] / df['beta_snn_acc_mean'])**2 + (base_beta_snn_acc_std / base_beta_snn_acc_mean)**2)**0.5

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

    # LSTM
    plt.plot(df_normalized['speed'], df_normalized['lstm_acc_mean'], label='LSTM', color=cmap(0.1), linestyle='-.')
    plt.fill_between(df_normalized['speed'], 
                     df_normalized['lstm_acc_mean'] - df_normalized['lstm_acc_std'], 
                     df_normalized['lstm_acc_mean'] + df_normalized['lstm_acc_std'], 
                     color=cmap(0.1), alpha=0.2)

    # CSNN
    plt.plot(df_normalized['speed'], df_normalized['csnn_acc_mean'], label='CSNN', color=cmap(0.5), linestyle='--')
    plt.fill_between(df_normalized['speed'], 
                     df_normalized['csnn_acc_mean'] - df_normalized['csnn_acc_std'], 
                     df_normalized['csnn_acc_mean'] + df_normalized['csnn_acc_std'], 
                     color=cmap(0.5), alpha=0.2)

    # Beta-SNN
    plt.plot(df_normalized['speed'], df_normalized['beta_snn_acc_mean'], label='Beta-SNN', color=cmap(0.9), linestyle='-')
    plt.fill_between(df_normalized['speed'], 
                     df_normalized['beta_snn_acc_mean'] - df_normalized['beta_snn_acc_std'], 
                     df_normalized['beta_snn_acc_mean'] + df_normalized['beta_snn_acc_std'], 
                     color=cmap(0.9), alpha=0.2)
    

    plt.xlabel('Speed')
    plt.ylabel('Normalized Accuracy')
    plt.ylim(0, 1)  # y軸の範囲を0から1に設定
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