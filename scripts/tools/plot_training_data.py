# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to plot training data from saved CSV files with beautiful visualizations."""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import seaborn as sns

# 设置中文字体和更好的样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_csv_data(data_dir: str):
    """从CSV文件加载时间序列数据。
    
    Args:
        data_dir: 训练数据保存目录
    
    Returns:
        DataFrame: 包含所有时间序列数据的DataFrame
    """
    # 查找所有CSV文件
    pattern = os.path.join(data_dir, "time_series_*.csv")
    files = sorted(glob.glob(pattern))
    
    if len(files) == 0:
        # 如果没有CSV文件，尝试加载NPZ文件（向后兼容）
        pattern = os.path.join(data_dir, "time_series_*.npz")
        files = sorted(glob.glob(pattern))
        if len(files) == 0:
            print(f"[警告] 没有找到时间序列数据文件")
            return None
        
        print(f"[信息] 找到 {len(files)} 个NPZ文件（向后兼容模式）")
        all_data = []
        for file in files:
            try:
                data = np.load(file)
                # 构建DataFrame，包含所有可能的字段
                df_dict = {'time': data['time']}
                for key in ['prediction_error', 'prediction_rmse', 'control_error', 'control_rmse',
                           'base_error_ratio', 'energy_consumption',
                           'platform_x', 'platform_y', 'platform_z',
                           'platform_roll', 'platform_pitch', 'platform_yaw',
                           'robot_x', 'robot_y', 'robot_z',
                           'robot_roll', 'robot_pitch', 'robot_yaw']:
                    if key in data:
                        df_dict[key] = data[key]
                df = pd.DataFrame(df_dict)
                all_data.append(df)
                print(f"[信息] 加载文件: {os.path.basename(file)} ({len(df)} 个数据点)")
            except Exception as e:
                print(f"[警告] 加载文件失败 {file}: {e}")
    else:
        print(f"[信息] 找到 {len(files)} 个CSV文件")
        all_data = []
        for file in files:
            try:
                df = pd.read_csv(file)
                all_data.append(df)
                print(f"[信息] 加载文件: {os.path.basename(file)} ({len(df)} 个数据点)")
            except Exception as e:
                print(f"[警告] 加载文件失败 {file}: {e}")
    
    if len(all_data) == 0:
        print("[错误] 没有有效的数据点")
        return None
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 按时间排序
    combined_df = combined_df.sort_values('time').reset_index(drop=True)
    
    print(f"[信息] 总共 {len(combined_df)} 个数据点")
    print(f"[信息] 时间范围: {combined_df['time'].min():.2f} - {combined_df['time'].max():.2f} 秒")
    
    return combined_df


def load_statistics(data_dir: str):
    """加载统计数据。
    
    Args:
        data_dir: 训练数据保存目录
    
    Returns:
        dict: 统计数据字典
    """
    # 优先使用JSON文件
    json_file = os.path.join(data_dir, "training_statistics.json")
    if os.path.exists(json_file):
        import json
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 其次使用CSV文件
    csv_file = os.path.join(data_dir, "training_statistics.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return dict(zip(df['metric'], df['value']))
    
    # 最后使用NPZ文件（向后兼容）
    npz_file = os.path.join(data_dir, "training_statistics.npz")
    if os.path.exists(npz_file):
        data = np.load(npz_file)
        return {key: float(data[key]) for key in data.keys()}
    
    return None


def plot_training_data(data_dir: str = "/home/user/IsaacLab/training_data"):
    """读取训练数据文件并绘制美观的图像。
    
    Args:
        data_dir: 训练数据保存目录
    """
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"[错误] 数据目录不存在: {data_dir}")
        return
    
    # 加载时间序列数据
    df = load_csv_data(data_dir)
    if df is None:
        return
    
    # 加载统计数据
    stats = load_statistics(data_dir)
    
    # 创建输出目录
    output_dir = data_dir
    
    # 计算移动平均窗口
    window = min(100, len(df) // 10) if len(df) > 10 else 1
    
    # 设置白色背景
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # ========== 图1: 平台预测误差RMSE随时间 ==========
    if 'prediction_rmse' in df.columns and df['prediction_rmse'].notna().any():
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
        ax.set_facecolor('white')
        
        ax.plot(df['time'], df['prediction_rmse'], 
               linewidth=2, color='#2E86AB', alpha=0.9, label='Prediction RMSE')
        
        if window > 1:
            df['pred_rmse_ma'] = df['prediction_rmse'].rolling(window=window, center=True).mean()
            ax.plot(df['time'], df['pred_rmse_ma'], 
                   linewidth=2.5, color='#1A5F7A', linestyle='--', alpha=0.8, label='Moving Average')
        
        ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
        ax.set_ylabel('RMSE (rad)', fontsize=13, fontweight='bold')
        ax.set_title('Platform Prediction Error RMSE Over Time', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, "prediction_rmse_over_time.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"[信息] 保存图像: {output_file}")
    else:
        print(f"[警告] 平台预测误差RMSE数据不存在")
    
    # ========== 图2: 平台预测误差RMSE和姿态误差RMSE的相关关系 ==========
    if ('prediction_rmse' in df.columns and 'control_rmse' in df.columns and 
        df['prediction_rmse'].notna().any() and df['control_rmse'].notna().any()):
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
        ax.set_facecolor('white')
        
        # 获取有效数据
        valid_mask = df['prediction_rmse'].notna() & df['control_rmse'].notna()
        pred_rmse_valid = df.loc[valid_mask, 'prediction_rmse']
        control_rmse_valid = df.loc[valid_mask, 'control_rmse']
        
        if valid_mask.sum() > 1:
            # 散点图
            ax.scatter(pred_rmse_valid, control_rmse_valid, 
                      alpha=0.6, s=30, color='#A23B72', edgecolors='none')
            
            # 检查数据是否有变化（不是常数）
            pred_std = pred_rmse_valid.std()
            control_std = control_rmse_valid.std()
            
            if pred_std > 1e-10 and control_std > 1e-10:
                try:
                    from scipy.stats import pearsonr
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        corr, p_value = pearsonr(pred_rmse_valid, control_rmse_valid)
                    
                    # 检查相关系数是否有效
                    if np.isnan(corr) or np.isinf(corr):
                        raise ValueError("Correlation coefficient is NaN or Inf")
                    
                    # 添加趋势线（使用try-except处理数值问题）
                    try:
                        z = np.polyfit(pred_rmse_valid, control_rmse_valid, 1)
                        # 检查拟合结果是否有效
                        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
                            raise ValueError("Polyfit result contains NaN or Inf")
                        p = np.poly1d(z)
                        x_fit = np.linspace(pred_rmse_valid.min(), pred_rmse_valid.max(), 100)
                        y_fit = p(x_fit)
                        # 检查拟合值是否有效
                        if np.any(np.isnan(y_fit)) or np.any(np.isinf(y_fit)):
                            raise ValueError("Fitted values contain NaN or Inf")
                        ax.plot(x_fit, y_fit, "r--", alpha=0.8, linewidth=2, 
                               label=f'Linear Fit (r={corr:.4f}, p={p_value:.4f})')
                    except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
                        # 如果polyfit失败，只显示相关系数，不画趋势线
                        print(f"[警告] 无法拟合趋势线: {e}")
                        ax.text(0.05, 0.95, f'Correlation: r={corr:.4f}, p={p_value:.4f}\n(Linear fit failed)', 
                               transform=ax.transAxes, fontsize=11, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except Exception as e:
                    print(f"[警告] 计算相关系数失败: {e}")
                    ax.text(0.05, 0.95, f'Correlation calculation failed: {str(e)[:50]}', 
                           transform=ax.transAxes, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            else:
                # 数据是常数，无法计算相关性
                if pred_std <= 1e-10:
                    print(f"[警告] prediction_rmse数据是常数（std={pred_std:.2e}），无法计算相关性")
                if control_std <= 1e-10:
                    print(f"[警告] control_rmse数据是常数（std={control_std:.2e}），无法计算相关性")
                ax.text(0.5, 0.5, 'Data is constant, cannot compute correlation', 
                       transform=ax.transAxes, fontsize=12, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('Prediction RMSE (rad)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Control RMSE (rad)', fontsize=13, fontweight='bold')
        ax.set_title('Correlation: Prediction RMSE vs Control RMSE', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')
        if ax.get_legend_handles_labels()[0]:  # 只有当有图例时才显示
            ax.legend(loc='best', fontsize=11, framealpha=0.95)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, "rmse_correlation.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"[信息] 保存图像: {output_file}")
    else:
        print(f"[警告] RMSE相关关系数据不完整")
    
    # ========== 图3: 强化学习训练的误差随时间变化（观察收敛） ==========
    if 'control_rmse' in df.columns and df['control_rmse'].notna().any():
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
        ax.set_facecolor('white')
        
        ax.plot(df['time'], df['control_rmse'], 
               linewidth=2, color='#8B008B', alpha=0.9, label='Control RMSE (Training Error)')
        
        if window > 1:
            df['control_rmse_ma'] = df['control_rmse'].rolling(window=window, center=True).mean()
            ax.plot(df['time'], df['control_rmse_ma'], 
                   linewidth=2.5, color='#6A0DAD', linestyle='--', alpha=0.8, label='Moving Average')
        
        # 计算最后100步的平均值，标记收敛点
        if len(df) >= 100:
            last_100_avg = df['control_rmse'].tail(100).mean()
            ax.axhline(y=last_100_avg, color='red', linestyle=':', linewidth=2, alpha=0.7, 
                      label=f'Last 100 steps avg: {last_100_avg:.6f} rad')
        
        ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
        ax.set_ylabel('RMSE (rad)', fontsize=13, fontweight='bold')
        ax.set_title('RL Policy Training Error Over Time (Convergence Analysis)', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, "rl_training_error.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"[信息] 保存图像: {output_file}")
    else:
        print(f"[警告] 强化学习训练误差数据不存在")
    
    # ========== 图4: 六自由度平台运动和机器狗基座运动（6张子图） ==========
    dof_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    dof_labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 
                  'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), facecolor='white')
    axes = axes.flatten()
    
    for i, (dof, label) in enumerate(zip(dof_names, dof_labels)):
        ax = axes[i]
        ax.set_facecolor('white')
        
        platform_col = f'platform_{dof}'
        robot_col = f'robot_{dof}'
        
        if platform_col in df.columns and robot_col in df.columns:
            if df[platform_col].notna().any() and df[robot_col].notna().any():
                ax.plot(df['time'], df[platform_col], 
                       linewidth=2, color='#FF6B6B', alpha=0.8, label='Platform', linestyle='-')
                ax.plot(df['time'], df[robot_col], 
                       linewidth=2, color='#4ECDC4', alpha=0.8, label='Robot Base', linestyle='-')
                
                if window > 1:
                    df[f'{platform_col}_ma'] = df[platform_col].rolling(window=window, center=True).mean()
                    df[f'{robot_col}_ma'] = df[robot_col].rolling(window=window, center=True).mean()
                    ax.plot(df['time'], df[f'{platform_col}_ma'], 
                           linewidth=2.5, color='#CC5555', linestyle='--', alpha=0.7, label='Platform (MA)')
                    ax.plot(df['time'], df[f'{robot_col}_ma'], 
                           linewidth=2.5, color='#3EAAA3', linestyle='--', alpha=0.7, label='Robot (MA)')
        
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'{label.split(" ")[0].capitalize()} Over Time', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax.legend(loc='best', fontsize=9, framealpha=0.95)
    
    plt.suptitle('Platform vs Robot Base 6-DOF Motion Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_file = os.path.join(output_dir, "6dof_motion_comparison.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[信息] 保存图像: {output_file}")
    
    # ========== 图5: 机器狗能量消耗随时间变化 ==========
    if 'energy_consumption' in df.columns and df['energy_consumption'].notna().any():
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
        ax.set_facecolor('white')
        
        ax.plot(df['time'], df['energy_consumption'], 
               linewidth=2, color='#F18F01', alpha=0.9, label='Energy Consumption')
        
        if window > 1:
            df['energy_ma'] = df['energy_consumption'].rolling(window=window, center=True).mean()
            ax.plot(df['time'], df['energy_ma'], 
                   linewidth=2.5, color='#C96F00', linestyle='--', alpha=0.8, label='Moving Average')
        
        # 添加平均线
        avg_energy = df['energy_consumption'].mean()
        ax.axhline(y=avg_energy, color='red', linestyle=':', linewidth=2, alpha=0.7, 
                  label=f'Average: {avg_energy:.2f}')
        
        ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Energy Consumption', fontsize=13, fontweight='bold')
        ax.set_title('Robot Energy Consumption Over Time', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, "energy_consumption.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"[信息] 保存图像: {output_file}")
    else:
        print(f"[警告] 能量消耗数据不存在")
    
    # ========== 统计信息图 ==========
    if stats:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
        
        # 子图1: 相关系数
        correlation = stats.get('prediction_control_correlation', 0)
        color_corr = '#d62728' if abs(correlation) < 0.1 else '#2ca02c'
        bars1 = axes[0].bar(['Correlation'], [correlation], color=color_corr, alpha=0.7, edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
        axes[0].set_title('Predictor vs Control Error\nCorrelation', fontsize=13, fontweight='bold', pad=10)
        axes[0].set_ylim([-1, 1])
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0].text(0, correlation, f'{correlation:.4f}', 
                    ha='center', va='bottom' if correlation >= 0 else 'top', 
                    fontsize=12, fontweight='bold')
        axes[0].set_facecolor('white')
        
        # 子图2: 平均存活时间
        avg_survival = stats.get('avg_survival_time', 0)
        bars2 = axes[1].bar(['Survival Time'], [avg_survival], color='#9467bd', alpha=0.7, edgecolor='black', linewidth=2)
        axes[1].set_ylabel('Steps', fontsize=12, fontweight='bold')
        axes[1].set_title('Average Robot\nSurvival Time', fontsize=13, fontweight='bold', pad=10)
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
        axes[1].text(0, avg_survival, f'{avg_survival:.1f}', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        axes[1].set_facecolor('white')
        
        # 子图3: 平均能量消耗
        avg_energy = stats.get('avg_energy_consumption', 0)
        bars3 = axes[2].bar(['Energy'], [avg_energy], color='#8c564b', alpha=0.7, edgecolor='black', linewidth=2)
        axes[2].set_ylabel('Energy Consumption', fontsize=12, fontweight='bold')
        axes[2].set_title('Average Energy\nConsumption', fontsize=13, fontweight='bold', pad=10)
        axes[2].grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
        axes[2].text(0, avg_energy, f'{avg_energy:.2f}', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        axes[2].set_facecolor('white')
        
        plt.suptitle('Training Statistics Summary', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        output_file = os.path.join(output_dir, "training_statistics.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"[信息] 保存图像: {output_file}")
        
        # 打印统计信息
        print("\n" + "="*60)
        print("训练统计数据:")
        print("="*60)
        print(f"预测器误差与控制误差的皮尔逊相关系数: {correlation:.6f}")
        print(f"各环境的平均机器狗存活时间: {avg_survival:.2f} 步")
        print(f"平均能量消耗: {avg_energy:.6f}")
        if 'total_energy' in stats:
            print(f"总能量消耗: {float(stats['total_energy']):.6f}")
        if 'total_time' in stats:
            print(f"总时间: {float(stats['total_time']):.2f} 秒")
        
        # 计算最后100步的平均误差
        if len(df) >= 100:
            if 'control_rmse' in df.columns and df['control_rmse'].notna().any():
                last_100_avg_error = df['control_rmse'].tail(100).mean()
                print(f"最后100步的平均误差: {last_100_avg_error:.6f} rad")
            if 'prediction_rmse' in df.columns and df['prediction_rmse'].notna().any():
                last_100_pred_rmse = df['prediction_rmse'].tail(100).mean()
                print(f"最后100步的平均预测RMSE: {last_100_pred_rmse:.6f} rad")
        
        print("="*60)
    else:
        print(f"[警告] 统计数据文件不存在")
    
    print(f"\n[完成] 所有图像已保存到: {data_dir}")


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="绘制训练数据图像（支持CSV和NPZ格式）")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/user/IsaacLab/training_data",
        help="训练数据保存目录（默认: /home/user/IsaacLab/training_data）"
    )
    
    args = parser.parse_args()
    
    plot_training_data(args.data_dir)


if __name__ == "__main__":
    main()
