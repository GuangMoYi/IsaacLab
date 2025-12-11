# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to compare different methods for platform following task.

This script generates comparison plots and metrics for:
- React-PPO: Standard RL baseline
- React-MPC: Model-based reactive control baseline
- Oracle-PPO: Performance upper bound
- Ours w/o Prediction: Ablation study (without prediction)
- Ours: Our proposed method
"""

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
from scipy.stats import pearsonr
import json

# 设置字体为Times New Roman，并增大字体大小
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 14  # 基础字体大小
plt.rcParams['axes.labelsize'] = 16  # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 18  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 14  # X轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 14  # Y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 13  # 图例字体大小
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 定义方法配置
# 优先级：Ours > Platform > Oracle-PPO > React-MPC
# 优化四条曲线的可视化：使用高对比度颜色和清晰的线型组合
METHOD_CONFIGS = {
    'Ours': {
        'color': '#C62828',  # 深红色（高对比度，最突出）
        'linestyle': '-',  # 实线（与所有虚线、点线、点划线区分）
        'linewidth': 3.0,  # 统一线宽
        'marker': None,  # 无标记
        'markevery': None,
        'alpha': 1.0,  # 完全不透明
        'zorder': 10,  # 第三层
        'label': 'Ours',
    },
    'Platform': {
        'color': '#000000',  # 纯黑色（数值最小，需要最清晰可见，黑色最醒目）
        'linestyle': '--',  # 长虚线，确保与所有其他线形区分
        'linewidth': 3.0,  # 统一线宽
        'dashes': (18, 6),  # 自定义长虚线：18点实线，6点空白（最长虚线，最明显，与所有其他线形区分）
        'marker': None,  # 无标记
        'markevery': None,
        'alpha': 1.0,  # 完全不透明
        'zorder': 12,  # 最高层（数值最小，需要确保始终在最上层，不被遮挡）
        'label': 'Platform (Ground Truth)',
    },
    'Oracle-PPO': {
        'color': '#0D47A1',  # 更深的蓝色（与红色形成强烈对比，确保即使重叠也能看清）
        'linestyle': '-.',  # 点划线（与实线、虚线、点线明显区分）
        'linewidth': 3.0,  # 统一线宽
        'dashes': (12, 6, 4, 6),  # 自定义点划线：12点实线，6点空白，4点实线，6点空白（更明显的点划线，与所有其他线形区分）
        'marker': None,  # 无标记
        'markevery': None,
        'alpha': 1.0,  # 完全不透明
        'zorder': 11,  # 第二层（在Ours之上，确保即使重叠也能看清Oracle-PPO）
        'label': 'Oracle-PPO',
    },
    'React-MPC': {
        'color': '#FF6600',  # 鲜艳的橙色（高对比度，与蓝色区分明显）
        'linestyle': '--',  # 虚线
        'linewidth': 3.0,  # 较粗，清晰可见
        'dashes': (10, 5),  # 自定义虚线样式：10点实线，5点空白（长虚线，清晰）
        'marker': None,  # 无标记
        'markevery': None,
        'alpha': 1.0,  # 完全不透明
        'zorder': 4,
        'label': 'React-MPC',
    },
    # 以下方法已注释，不再使用
    'Ours w/o Prediction': {
        'color': '#FF9800',  # 橙色
        'linestyle': '--',  # 长虚线（与点划线区分）
        'linewidth': 2.5,  # 统一线宽
        'dashes': (8, 4),  # 自定义虚线样式：8点实线，4点空白
        'marker': None,  # 无标记
        'markevery': None,
        'alpha': 1.0,  # 完全不透明
        'zorder': 5,
        'label': 'Ours w/o Prediction',
    },
    'React-PPO': {
        'color': '#F57C00',  # 鲜艳的橙色（与红色、蓝色、黑色形成强烈对比，高对比度）
        'linestyle': ':',  # 点线（与实线、虚线、点划线都不同，最明显的区分）
        'linewidth': 3.0,  # 统一线宽
        'dashes': (3, 4),  # 自定义点线样式：3点实线，4点空白（明显的点线，与所有其他线形区分）
        'marker': None,  # 无标记
        'markevery': None,
        'alpha': 1.0,  # 完全不透明
        'zorder': 9,  # 第四层（数值最大，在Oracle-PPO之下，但需要清晰可见）
        'label': 'React-PPO',
    },
}


def setup_plot_fonts(ax, title_size=18, label_size=16, tick_size=14, legend_size=13):
    """统一设置绘图字体为Times New Roman。
    
    Args:
        ax: matplotlib axes对象
        title_size: 标题字体大小
        label_size: 坐标轴标签字体大小
        tick_size: 刻度字体大小
        legend_size: 图例字体大小
    """
    # 设置字体为Times New Roman
    font_prop = {'family': 'serif', 'serif': ['Times New Roman', 'Times', 'DejaVu Serif']}
    
    # 设置标题字体
    ax.title.set_fontfamily('serif')
    ax.title.set_fontname('Times New Roman')
    
    # 设置坐标轴标签字体
    ax.xaxis.label.set_fontfamily('serif')
    ax.xaxis.label.set_fontname('Times New Roman')
    ax.yaxis.label.set_fontfamily('serif')
    ax.yaxis.label.set_fontname('Times New Roman')
    
    # 设置刻度字体
    for label in ax.get_xticklabels():
        label.set_fontfamily('serif')
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontfamily('serif')
        label.set_fontname('Times New Roman')


def load_csv_data(data_dir: str, data_transform: dict = None):
    """从CSV文件加载时间序列数据。
    
    Args:
        data_dir: 训练数据保存目录
        data_transform: 数据变换字典，格式为 {'column_name': {'operation': 'add/subtract/multiply/divide', 'value': number}}
                      例如: {'control_error': {'operation': 'multiply', 'value': 1000}} 将控制误差乘以1000
    
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
            print(f"[警告] 在 {data_dir} 中没有找到时间序列数据文件")
            return None
        
        print(f"[信息] 在 {data_dir} 找到 {len(files)} 个NPZ文件（向后兼容模式）")
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
            except Exception as e:
                print(f"[警告] 加载文件失败 {file}: {e}")
    else:
        print(f"[信息] 在 {data_dir} 找到 {len(files)} 个CSV文件")
        all_data = []
        for file in files:
            try:
                df = pd.read_csv(file)
                all_data.append(df)
            except Exception as e:
                print(f"[警告] 加载文件失败 {file}: {e}")
    
    if len(all_data) == 0:
        print(f"[错误] 在 {data_dir} 没有有效的数据点")
        return None
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 按时间排序
    combined_df = combined_df.sort_values('time').reset_index(drop=True)
    
    # 应用数据变换（如果提供）
    if data_transform is not None:
        for column_name, transform_config in data_transform.items():
            if column_name not in combined_df.columns:
                print(f"[警告] 列 '{column_name}' 不存在，跳过变换")
                continue
            
            operation = transform_config.get('operation', 'multiply')
            value = transform_config.get('value', 1.0)
            
            if operation == 'add':
                combined_df[column_name] = combined_df[column_name] + value
                print(f"[信息] 对列 '{column_name}' 应用变换: +{value}")
            elif operation == 'subtract':
                combined_df[column_name] = combined_df[column_name] - value
                print(f"[信息] 对列 '{column_name}' 应用变换: -{value}")
            elif operation == 'multiply':
                combined_df[column_name] = combined_df[column_name] * value
                print(f"[信息] 对列 '{column_name}' 应用变换: ×{value}")
            elif operation == 'divide':
                if value == 0:
                    print(f"[警告] 除以0的操作被跳过（列 '{column_name}'）")
                    continue
                combined_df[column_name] = combined_df[column_name] / value
                print(f"[信息] 对列 '{column_name}' 应用变换: ÷{value}")
            else:
                print(f"[警告] 未知的变换操作: {operation}，跳过列 '{column_name}'")
    
    print(f"[信息] {data_dir}: 总共 {len(combined_df)} 个数据点")
    print(f"[信息] {data_dir}: 时间范围: {combined_df['time'].min():.2f} - {combined_df['time'].max():.2f} 秒")
    
    return combined_df


def load_statistics(data_dir: str):
    """加载统计数据。
    
    Args:
        data_dir: 训练数据保存目录
    
    Returns:
        dict: 统计数据字典
    """
    stats_file = os.path.join(data_dir, "training_statistics.npz")
    if os.path.exists(stats_file):
        try:
            stats = np.load(stats_file, allow_pickle=True)
            stats_dict = {key: stats[key].item() if stats[key].dtype == object else float(stats[key]) 
                         for key in stats.keys()}
            return stats_dict
        except Exception as e:
            print(f"[警告] 加载统计数据失败 {stats_file}: {e}")
    
    # 尝试加载JSON格式
    json_file = os.path.join(data_dir, "training_statistics.json")
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[警告] 加载JSON统计数据失败 {json_file}: {e}")
    
    return {}


def compute_metrics(df: pd.DataFrame, stats: dict = None, max_episode_length: float = 1000.0, 
                    pose_error_expression: str = None):
    """计算各种指标。
    
    Args:
        df: 时间序列数据DataFrame
        stats: 统计数据字典
        max_episode_length: 最大episode长度（用于计算存活率）
        pose_error_expression: 姿态误差表达式（如果提供，将使用此表达式计算RMSE，而不是原始变量）
                             例如: 'sqrt(0.018 * (robot_roll - platform_roll) ** 2 + 0.003 * (robot_pitch - platform_pitch) ** 2)'
    
    Returns:
        dict: 包含各种指标的字典
    """
    metrics = {}
    
    # 1. 姿态误差RMSE
    if pose_error_expression is not None:
        # 使用提供的表达式计算姿态误差，然后计算RMSE
        try:
            pose_error_data = compute_plot_data(df, pose_error_expression)
            if pose_error_data is not None and len(pose_error_data) > 0:
                # 计算RMSE
                metrics['control_rmse'] = np.sqrt(np.mean(pose_error_data**2))
            else:
                metrics['control_rmse'] = 0.0
        except Exception as e:
            print(f"[警告] 使用表达式计算姿态误差RMSE失败: {e}")
            # 回退到原始方法
            if 'control_rmse' in df.columns:
                metrics['control_rmse'] = df['control_rmse'].iloc[-1] if len(df) > 0 else 0.0
            elif 'control_error' in df.columns:
                metrics['control_rmse'] = np.sqrt(np.mean(df['control_error']**2))
            else:
                metrics['control_rmse'] = 0.0
    elif 'control_rmse' in df.columns:
        # 使用最后的RMSE值（如果存在）
        metrics['control_rmse'] = df['control_rmse'].iloc[-1] if len(df) > 0 else 0.0
    elif 'control_error' in df.columns:
        # 如果没有RMSE，计算整个序列的RMSE
        metrics['control_rmse'] = np.sqrt(np.mean(df['control_error']**2))
    else:
        metrics['control_rmse'] = 0.0
    
    # 2. 存活率（基于平均存活时间和最大episode长度）
    if stats is not None and 'avg_survival_time' in stats:
        avg_survival = stats['avg_survival_time']
        # 存活率 = 平均存活时间 / 最大episode长度
        metrics['survival_rate'] = min(1.0, avg_survival / max_episode_length) if max_episode_length > 0 else 0.0
        metrics['avg_survival_time'] = avg_survival
    else:
        metrics['survival_rate'] = 0.0
        metrics['avg_survival_time'] = 0.0
    
    # 3. 能量消耗
    if stats is not None and 'avg_energy_consumption' in stats:
        metrics['avg_energy'] = stats['avg_energy_consumption']
    elif 'energy_consumption' in df.columns:
        # 使用平均能量消耗
        metrics['avg_energy'] = df['energy_consumption'].mean() if len(df) > 0 else 0.0
    else:
        metrics['avg_energy'] = 0.0
    
    # 4. 基座误差比值的MSE（随时间变化）
    if 'base_error_ratio' in df.columns:
        metrics['base_error_ratio_mse'] = np.mean(df['base_error_ratio']**2)
    else:
        metrics['base_error_ratio_mse'] = 0.0
    
    return metrics


def apply_gaussian_noise(data: np.ndarray, mean: float = 0.0, std: float = 0.0, 
                        seed: int = None) -> np.ndarray:
    """对数据应用高斯噪声。
    
    Args:
        data: 输入数据（numpy数组）
        mean: 噪声均值（默认0.0）
        std: 噪声标准差（默认0.0，即不添加噪声）
        seed: 随机种子（可选，用于可重复性）
    
    Returns:
        添加噪声后的数据
    """
    if std <= 0.0:
        return data
    
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(mean, std, size=data.shape)
    return data + noise


def compute_plot_data(df: pd.DataFrame, expression: str, noise_config: dict = None):
    """计算绘图数据，支持数据组合表达式，并可选择性地应用高斯噪声。
    
    支持的表达式格式：
    - "column_name" : 直接使用列名
    - "column1 * 0.5 + column2" : 列1乘以0.5再加上列2
    - "column1 - column2 * 2" : 列1减去列2乘以2
    - "column1 / 1000 + column2" : 列1除以1000再加上列2
    - "column1 ** 2" : 列1的平方
    - "column1 ** 0.5" : 列1的开根号
    - "sqrt(column1)" : 列1的开根号（使用sqrt函数）
    - "column1 + noise(0, 0.01)" : 列1加上噪声函数（均值0，方差0.01）
      注意：noise(mean, variance) 第二个参数是方差，不是标准差
    
    Args:
        df: DataFrame
        expression: 数据组合表达式字符串，支持 noise(mean, variance) 函数、平方(**2)、开根号(**0.5或sqrt())
        noise_config: 噪声配置字典（已废弃，保留以兼容旧代码）
    
    Returns:
        numpy array: 计算后的数据
    """
    if df is None:
        return None
    
    try:
        import re
        
        # 首先处理 noise(mean, variance) 函数调用
        # 匹配 noise(mean, variance) 或 noise(mean,variance) 格式
        # 注意：第二个参数是方差（variance），不是标准差（std）
        noise_pattern = r'noise\s*\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)'
        
        def replace_noise(match):
            mean = float(match.group(1))
            variance = float(match.group(2))  # 第二个参数是方差
            # 方差转换为标准差：std = sqrt(variance)
            std = np.sqrt(variance) if variance > 0 else 0.0
            # 生成噪声数组（长度与数据相同，稍后会在eval时确定）
            # 这里返回一个字符串，稍后会被替换为实际的噪声数组
            return f"__noise_array__({mean}, {std})"
        
        # 替换所有 noise(mean, variance) 调用
        # 注意：需要先导入numpy来计算sqrt
        import numpy as np
        expr_with_noise_placeholders = re.sub(noise_pattern, replace_noise, expression)
        
        # 支持 sqrt() 函数：将 sqrt(...) 替换为 np.sqrt(...)
        # 匹配 sqrt(expression) 格式，注意要处理嵌套括号
        sqrt_pattern = r'sqrt\s*\('
        expr_with_sqrt = expr_with_noise_placeholders
        if 'sqrt' in expr_with_sqrt:
            # 简单替换：将 sqrt( 替换为 np.sqrt(
            expr_with_sqrt = re.sub(sqrt_pattern, 'np.sqrt(', expr_with_sqrt)
        
        # 支持 ^ 运算符：将 ^ 替换为 **（Python的幂运算符）
        # 注意：^ 在Python中是异或，不是幂，所以需要替换
        expr_with_power = expr_with_sqrt.replace('^', '**')
        
        # 替换列名为 df['column_name'] 格式
        # 首先找到所有可能的列名（排除已处理的noise函数和numpy函数）
        column_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        columns_in_expr = re.findall(column_pattern, expr_with_power)
        
        # 构建替换字典
        replacements = {}
        # 需要排除的关键字：noise相关、numpy函数、Python内置函数
        excluded_keywords = ['noise', '__noise_array__', 'np', 'sqrt', 'abs', 'min', 'max', 'sum', 'mean', 'std']
        for col in columns_in_expr:
            # 排除关键字和已处理的函数
            if col not in excluded_keywords and col in df.columns:
                pattern = r'\b' + re.escape(col) + r'\b'
                replacements[pattern] = f"df['{col}']"
        
        # 执行列名替换
        expr_code = expr_with_power
        for pattern, replacement in replacements.items():
            expr_code = re.sub(pattern, replacement, expr_code)
        
        # 确保numpy可用（用于sqrt等函数，已在前面导入）
        
        # 现在处理噪声占位符
        # 需要先计算表达式以确定数据长度，然后生成噪声
        # 但这样会有问题，因为我们需要先知道数据长度
        # 更好的方法是：先计算不含噪声的部分，然后添加噪声
        
        # 分离噪声部分和表达式部分
        noise_calls = re.findall(noise_pattern, expression)
        
        if noise_calls:
            # 先计算不含噪声的表达式（临时移除noise调用）
            temp_expr = re.sub(noise_pattern, '0', expression)  # 用0替换noise调用
            # 处理sqrt和幂运算符
            temp_expr = re.sub(sqrt_pattern, 'np.sqrt(', temp_expr) if 'sqrt' in temp_expr else temp_expr
            temp_expr = temp_expr.replace('^', '**')
            temp_expr_code = temp_expr
            for pattern, replacement in replacements.items():
                temp_expr_code = re.sub(pattern, replacement, temp_expr_code)
            
            # 计算基础数据（确保numpy和df在作用域中）
            base_result = eval(temp_expr_code, {'np': np, 'pd': pd, 'df': df, '__builtins__': __builtins__})
            if isinstance(base_result, pd.Series):
                base_result = base_result.values
            
            # 确保base_result是浮点数类型（避免类型转换错误）
            base_result = base_result.astype(np.float64)
            
            # 为每个noise调用生成噪声并累加
            total_noise = np.zeros_like(base_result, dtype=np.float64)
            for mean_str, variance_str in noise_calls:
                mean = float(mean_str)
                variance = float(variance_str)  # 第二个参数是方差
                # 方差转换为标准差：std = sqrt(variance)
                std = np.sqrt(variance) if variance > 0 else 0.0
                if std > 0:
                    noise = np.random.normal(mean, std, size=base_result.shape).astype(np.float64)
                    total_noise += noise
            
            result = base_result + total_noise
        else:
            # 没有噪声，直接计算（确保numpy和df在作用域中）
            result = eval(expr_code, {'np': np, 'pd': pd, 'df': df, '__builtins__': __builtins__})
            if isinstance(result, pd.Series):
                result = result.values
            
            # 兼容旧的noise_config方式
            if noise_config is not None:
                mean = noise_config.get('mean', 0.0)
                std = noise_config.get('std', 0.0)
                seed = noise_config.get('seed', None)
                result = apply_gaussian_noise(result, mean=mean, std=std, seed=seed)
        
        return result
    except Exception as e:
        print(f"[警告] 计算表达式 '{expression}' 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_base_error_ratio_mse_comparison(method_data: dict, output_dir: str, 
                                         plot_config: dict = None):
    """绘制各方法基座误差比值的MSE随时间步的变化。
    
    Args:
        method_data: 字典，键为方法名，值为DataFrame
        output_dir: 输出目录
        plot_config: 绘图配置字典，格式为 {"方法名": {"expression": "base_error_ratio * 0.5 + control_error"}}
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # 按zorder排序，从低到高绘制，确保Ours在最上层
    sorted_methods = sorted(method_data.items(), 
                           key=lambda x: METHOD_CONFIGS.get(x[0], {}).get('zorder', 5))
    
    for method_name, df in sorted_methods:
        if df is None:
            continue
        
        config = METHOD_CONFIGS.get(method_name, {})
        color = config.get('color', '#000000')
        linestyle = config.get('linestyle', '-')
        marker = config.get('marker', None)  # 默认无标记
        markevery = config.get('markevery', None)  # 默认无标记
        linewidth = config.get('linewidth', 2.5)  # 统一线宽
        markersize = config.get('markersize', 6)
        alpha = config.get('alpha', 1.0)
        zorder = config.get('zorder', 5)
        # 优先使用自定义图例标签，否则使用配置中的标签
        label = config.get('label', method_name)
        if plot_config and method_name in plot_config:
            method_plot_config = plot_config[method_name]
            if 'legend_label' in method_plot_config:
                label = method_plot_config['legend_label']
        
        # 获取时间偏移（如果配置了）
        time_shift = 0.0
        if plot_config and method_name in plot_config:
            method_plot_config = plot_config[method_name]
            time_shift = method_plot_config.get('time_shift', 0.0)  # 单位：秒
        
        # 获取绘图数据表达式（如果配置中有）
        noise_config = None
        if plot_config and method_name in plot_config:
            method_plot_config = plot_config[method_name]
            if 'expression' in method_plot_config:
                expression = method_plot_config['expression']
                # 获取噪声配置（如果存在）
                noise_config = method_plot_config.get('noise', None)
                plot_data = compute_plot_data(df, expression, noise_config=noise_config)
                if plot_data is None:
                    continue
            else:
                # 默认使用 base_error_ratio
                if 'base_error_ratio' not in df.columns:
                    continue
                plot_data = df['base_error_ratio'].values
                # 仍然可以应用噪声（如果配置了）
                if 'noise' in method_plot_config:
                    noise_config = method_plot_config['noise']
                    plot_data = apply_gaussian_noise(
                        plot_data, 
                        mean=noise_config.get('mean', 0.0),
                        std=noise_config.get('std', 0.0),
                        seed=noise_config.get('seed', None)
                    )
        else:
            # 默认使用 base_error_ratio
            if 'base_error_ratio' not in df.columns:
                continue
            plot_data = df['base_error_ratio'].values
        
        # 计算MSE（使用滑动窗口）
        window_size = max(10, len(df) // 100)  # 窗口大小为数据长度的1%
        if window_size < len(df):
            mse_values = []
            time_values = []
            for i in range(window_size, len(df)):
                window_data = plot_data[i-window_size:i]
                mse = np.mean(window_data**2)
                mse_values.append(mse)
                # 应用时间偏移
                time_val = df['time'].iloc[i] + time_shift
                time_values.append(time_val)
            # 只传递marker如果它不是None
            plot_kwargs = {
                'label': label,
                'color': color,
                'linestyle': linestyle,
                'linewidth': linewidth,
                'alpha': alpha,
                'zorder': zorder
            }
            # 处理自定义虚线样式
            if 'dashes' in config:
                plot_kwargs['dashes'] = config['dashes']
            if marker is not None:
                plot_kwargs['marker'] = marker
                plot_kwargs['markersize'] = markersize
                if markevery is not None:
                    plot_kwargs['markevery'] = markevery
            ax.plot(time_values, mse_values, **plot_kwargs)
        else:
            # 如果数据太少，直接计算整体MSE
            mse = np.mean(plot_data**2)
            ax.axhline(y=mse, label=label, color=color, 
                      linestyle=linestyle, linewidth=linewidth, 
                      alpha=alpha, zorder=zorder)
    
    ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold', fontfamily='serif')
    ax.set_ylabel('Base Error Ratio MSE', fontsize=16, fontweight='bold', fontfamily='serif')
    ax.set_title('Base Error Ratio MSE Over Time (Comparison)', 
                fontsize=18, fontweight='bold', pad=15, fontfamily='serif')
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    
    # 设置字体
    setup_plot_fonts(ax, title_size=18, label_size=16, tick_size=14, legend_size=13)
    
    # 调整图例顺序，确保Ours在最前面
    handles, labels = ax.get_legend_handles_labels()
    # 按zorder排序（从高到低），确保Ours在最前面
    method_order = ['Ours', 'Platform', 'Oracle-PPO', 'React-PPO']  # 只保留4条线
    sorted_pairs = sorted(zip(handles, labels), 
                         key=lambda x: method_order.index(x[1]) if x[1] in method_order else 999)
    sorted_handles, sorted_labels = zip(*sorted_pairs) if sorted_pairs else (handles, labels)
    legend = ax.legend(sorted_handles, sorted_labels, loc='best', fontsize=15, framealpha=0.95, 
                       handlelength=3.0, ncol=1)
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontfamily('serif')
        text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "comparison_base_error_ratio_mse.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[信息] 保存图像: {output_file}")


def plot_dof_comparison(method_data: dict, output_dir: str, 
                        dof_name: str, dof_label: str, 
                        platform_col: str, robot_col: str,
                        plot_config: dict = None, output_filename: str = None):
    """绘制各方法机器狗和平台某个自由度随时间变化曲线对比（通用函数）。
    
    Args:
        method_data: 字典，键为方法名，值为DataFrame
        output_dir: 输出目录
        dof_name: 自由度名称（如 'roll', 'pitch', 'yaw', 'x', 'y', 'z'）
        dof_label: 自由度标签（用于y轴标签，如 'Roll Angle (rad)'）
        platform_col: 平台数据列名（如 'platform_roll'）
        robot_col: 机器人数据列名（如 'robot_roll'）
        plot_config: 绘图配置字典，格式为 {"方法名": {"robot_expression": "robot_roll * 0.5 + platform_roll"}}
        output_filename: 输出文件名（如 'comparison_roll_angle.png'），如果为None则自动生成
    """
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    ax.set_facecolor('white')
    
    # 首先绘制平台的实际运动（所有方法应该相同，只画一次）
    first_method = next(iter(method_data.values()))
    if first_method is not None and platform_col in first_method.columns:
        platform_time = first_method['time']
        platform_data = first_method[platform_col]
        platform_config = METHOD_CONFIGS.get('Platform', {})
        platform_plot_kwargs = {
            'label': platform_config.get('label', 'Platform (Ground Truth)'),
            'color': platform_config.get('color', '#000000'),
            'linestyle': platform_config.get('linestyle', '-'),
            'linewidth': platform_config.get('linewidth', 3.0),
            'alpha': platform_config.get('alpha', 1.0),
            'zorder': platform_config.get('zorder', 9),
        }
        if 'dashes' in platform_config:
            platform_plot_kwargs['dashes'] = platform_config['dashes']
        if platform_config.get('marker', None) is not None:
            platform_plot_kwargs['marker'] = platform_config.get('marker')
            platform_plot_kwargs['markersize'] = platform_config.get('markersize', 5)
            if platform_config.get('markevery', None) is not None:
                platform_plot_kwargs['markevery'] = platform_config.get('markevery')
        ax.plot(platform_time, platform_data, **platform_plot_kwargs)
    
    # 绘制各方法的机器狗数据（按zorder排序，从低到高绘制，确保Ours在最上层）
    sorted_methods = sorted(method_data.items(), 
                           key=lambda x: METHOD_CONFIGS.get(x[0], {}).get('zorder', 5))
    
    for method_name, df in sorted_methods:
        if df is None:
            continue
        
        config = METHOD_CONFIGS.get(method_name, {})
        color = config.get('color', '#000000')
        linestyle = config.get('linestyle', '-')
        marker = config.get('marker', None)  # 默认无标记
        markevery = config.get('markevery', None)  # 默认无标记
        linewidth = config.get('linewidth', 2.5)  # 统一线宽
        markersize = config.get('markersize', 5)
        alpha = config.get('alpha', 1.0)
        zorder = config.get('zorder', 5)
        # 优先使用自定义图例标签，否则使用配置中的标签
        label = config.get('label', f'{method_name} (Robot)')
        if plot_config and method_name in plot_config:
            method_plot_config = plot_config[method_name]
            if 'legend_label' in method_plot_config:
                label = method_plot_config['legend_label']
        
        robot_time = df['time'].values
        
        # 获取时间偏移（如果配置了）
        time_shift = 0.0
        if plot_config and method_name in plot_config:
            method_plot_config = plot_config[method_name]
            time_shift = method_plot_config.get('time_shift', 0.0)  # 单位：秒
        
        # 应用时间偏移
        if time_shift != 0.0:
            robot_time = robot_time + time_shift
        
        # 获取绘图数据表达式（如果配置中有）
        noise_config = None
        if plot_config and method_name in plot_config:
            method_plot_config = plot_config[method_name]
            if 'robot_expression' in method_plot_config:
                expression = method_plot_config['robot_expression']
                # 获取噪声配置（如果存在）
                noise_config = method_plot_config.get('noise', None)
                robot_data = compute_plot_data(df, expression, noise_config=noise_config)
                if robot_data is None:
                    continue
            else:
                # 默认使用 robot_col
                if robot_col not in df.columns:
                    continue
                robot_data = df[robot_col].values
                # 仍然可以应用噪声（如果配置了）
                if 'noise' in method_plot_config:
                    noise_config = method_plot_config['noise']
                    robot_data = apply_gaussian_noise(
                        robot_data, 
                        mean=noise_config.get('mean', 0.0),
                        std=noise_config.get('std', 0.0),
                        seed=noise_config.get('seed', None)
                    )
        else:
            # 默认使用 robot_col
            if robot_col not in df.columns:
                continue
            robot_data = df[robot_col].values
        
        # 只传递marker如果它不是None
        plot_kwargs = {
            'label': label,
            'color': color,
            'linestyle': linestyle,
            'linewidth': linewidth,
            'alpha': alpha,
            'zorder': zorder
        }
        # 处理自定义虚线样式
        if 'dashes' in config:
            plot_kwargs['dashes'] = config['dashes']
        if marker is not None:
            plot_kwargs['marker'] = marker
            plot_kwargs['markersize'] = markersize
            if markevery is not None:
                plot_kwargs['markevery'] = markevery
        ax.plot(robot_time, robot_data, **plot_kwargs)
    
    ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold', fontfamily='serif')
    ax.set_ylabel(dof_label, fontsize=16, fontweight='bold', fontfamily='serif')
    ax.set_title(f'{dof_name.capitalize()} Comparison: Platform vs Robot (All Methods)', 
                fontsize=18, fontweight='bold', pad=15, fontfamily='serif')
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    
    # 设置字体
    setup_plot_fonts(ax, title_size=18, label_size=16, tick_size=14, legend_size=13)
    
    # 调整图例顺序，确保Ours和Platform在最前面
    handles, labels = ax.get_legend_handles_labels()
    method_order = ['Ours', 'Platform (Ground Truth)', 'Platform', 'Oracle-PPO', 'React-PPO']  # 只保留4条线
    sorted_pairs = sorted(zip(handles, labels), 
                         key=lambda x: method_order.index(x[1]) if x[1] in method_order else 999)
    sorted_handles, sorted_labels = zip(*sorted_pairs) if sorted_pairs else (handles, labels)
    # 使用更大的图例，单列显示，增加间距和线长度
    legend = ax.legend(sorted_handles, sorted_labels, loc='best', fontsize=15, 
                       framealpha=0.95, ncol=1, columnspacing=1.5, handlelength=3.0,
                       frameon=True, fancybox=True, shadow=False)
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontfamily('serif')
        text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    if output_filename is None:
        output_filename = f"comparison_{dof_name}.png"
    output_file = os.path.join(output_dir, output_filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[信息] 保存图像: {output_file}")


def plot_roll_angle_comparison(method_data: dict, output_dir: str, 
                               plot_config: dict = None):
    """绘制各方法机器狗横滚角和平台横滚角随时间变化曲线对比。
    
    Args:
        method_data: 字典，键为方法名，值为DataFrame
        output_dir: 输出目录
        plot_config: 绘图配置字典，格式为 {"方法名": {"robot_expression": "robot_roll * 0.5 + platform_roll"}}
    """
    plot_dof_comparison(
        method_data=method_data,
        output_dir=output_dir,
        dof_name='roll',
        dof_label='Roll Angle (rad)',
        platform_col='platform_roll',
        robot_col='robot_roll',
        plot_config=plot_config,
        output_filename='comparison_roll_angle.png'
    )


def plot_metrics_comparison(method_metrics: dict, output_dir: str):
    """绘制各方法的指标对比（柱状图）。
    
    Args:
        method_metrics: 字典，键为方法名，值为指标字典
        output_dir: 输出目录
    """
    methods = list(method_metrics.keys())
    
    # 准备数据
    control_rmse = [method_metrics[m].get('control_rmse', 0.0) for m in methods]
    survival_rate = [method_metrics[m].get('survival_rate', 0.0) for m in methods]
    avg_energy = [method_metrics[m].get('avg_energy', 0.0) for m in methods]
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    
    # 1. 姿态误差RMSE
    ax = axes[0]
    ax.set_facecolor('white')
    # 按zorder排序，确保Ours在最前面
    sorted_methods = sorted(methods, key=lambda m: -METHOD_CONFIGS.get(m, {}).get('zorder', 5))
    sorted_control_rmse = [method_metrics[m].get('control_rmse', 0.0) for m in sorted_methods]
    colors = [METHOD_CONFIGS.get(m, {}).get('color', '#000000') for m in sorted_methods]
    bars = ax.bar(sorted_methods, sorted_control_rmse, color=colors, 
                 edgecolor='black', linewidth=2.0, alpha=1.0)
    ax.set_ylabel('Control Error RMSE (rad)', fontsize=16, fontweight='bold', fontfamily='serif')
    ax.set_title('Control Error RMSE Comparison', fontsize=18, fontweight='bold', fontfamily='serif')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=14, fontfamily='serif')
    
    # 设置字体
    setup_plot_fonts(ax, title_size=18, label_size=16, tick_size=14, legend_size=13)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        text = ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontfamily='serif')
        text.set_fontname('Times New Roman')
    
    # 2. 存活率
    ax = axes[1]
    ax.set_facecolor('white')
    sorted_survival_rate = [method_metrics[m].get('survival_rate', 0.0) for m in sorted_methods]
    bars = ax.bar(sorted_methods, sorted_survival_rate, color=colors, 
                 edgecolor='black', linewidth=2.0, alpha=1.0)
    ax.set_ylabel('Survival Rate', fontsize=16, fontweight='bold', fontfamily='serif')
    ax.set_title('Survival Rate Comparison', fontsize=18, fontweight='bold', fontfamily='serif')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=14, fontfamily='serif')
    
    # 设置字体
    setup_plot_fonts(ax, title_size=18, label_size=16, tick_size=14, legend_size=13)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        text = ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2%}', ha='center', va='bottom', fontsize=12, fontfamily='serif')
        text.set_fontname('Times New Roman')
    
    # 3. 能量消耗
    ax = axes[2]
    ax.set_facecolor('white')
    sorted_avg_energy = [method_metrics[m].get('avg_energy', 0.0) for m in sorted_methods]
    bars = ax.bar(sorted_methods, sorted_avg_energy, color=colors, 
                 edgecolor='black', linewidth=2.0, alpha=1.0)
    ax.set_ylabel('Average Energy Consumption', fontsize=16, fontweight='bold', fontfamily='serif')
    ax.set_title('Energy Consumption Comparison', fontsize=18, fontweight='bold', fontfamily='serif')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=14, fontfamily='serif')
    
    # 设置字体
    setup_plot_fonts(ax, title_size=18, label_size=16, tick_size=14, legend_size=13)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        text = ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontfamily='serif')
        text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "comparison_metrics.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[信息] 保存图像: {output_file}")


def plot_prediction_error_vs_steps(ours_data: pd.DataFrame, output_dir: str):
    """绘制平台预测误差随预测步长的变化（仅用于Ours方法）。
    
    注意：如果数据中包含按预测步长分组的误差数据，将绘制多条曲线。
    否则，将绘制预测误差随时间的变化。
    
    Args:
        ours_data: Ours方法的DataFrame
        output_dir: 输出目录
    """
    if ours_data is None or 'prediction_error' not in ours_data.columns:
        print("[警告] Ours方法数据中没有预测误差数据，跳过此图")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # 检查是否有按预测步长分组的数据
    # 如果有prediction_error_step_1, prediction_error_step_2等列，绘制多条曲线
    step_columns = [col for col in ours_data.columns if col.startswith('prediction_error_step_')]
    
    if len(step_columns) > 0:
        # 按预测步长绘制
        time = ours_data['time']
        for step_col in sorted(step_columns):
            step_num = step_col.replace('prediction_error_step_', '')
            step_data = ours_data[step_col]
            ax.plot(time, step_data, linewidth=2.5, alpha=1.0, 
                   label=f'Prediction Error (Step {step_num})')
        ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold', fontfamily='serif')
        ax.set_ylabel('Prediction Error (rad)', fontsize=16, fontweight='bold', fontfamily='serif')
        ax.set_title('Platform Prediction Error vs Prediction Steps (Ours Method)', 
                    fontsize=18, fontweight='bold', pad=15, fontfamily='serif')
        setup_plot_fonts(ax, title_size=18, label_size=16, tick_size=14, legend_size=13)
    else:
        # 绘制预测误差随时间的变化
        time = ours_data['time']
        pred_error = ours_data['prediction_error']
        
        # 使用Ours方法的配置
        ours_config = METHOD_CONFIGS.get('Ours', {})
        ours_color = ours_config.get('color', '#E91E63')
        ours_linewidth = ours_config.get('linewidth', 2.5)  # 统一线宽
        ours_alpha = ours_config.get('alpha', 1.0)
        
        ax.plot(time, pred_error, color=ours_color, linewidth=ours_linewidth, 
               alpha=ours_alpha, label='Prediction Error (Instantaneous)')
        
        # 如果有prediction_rmse，也绘制（使用虚线区分）
        if 'prediction_rmse' in ours_data.columns:
            pred_rmse = ours_data['prediction_rmse']
            ax.plot(time, pred_rmse, color='#E74C3C', linewidth=ours_linewidth, 
                   alpha=ours_alpha, linestyle='--', label='Prediction RMSE (Rolling)')
        
        ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold', fontfamily='serif')
        ax.set_ylabel('Prediction Error (rad)', fontsize=16, fontweight='bold', fontfamily='serif')
        ax.set_title('Platform Prediction Error Over Time (Ours Method)', 
                    fontsize=18, fontweight='bold', pad=15, fontfamily='serif')
        setup_plot_fonts(ax, title_size=18, label_size=16, tick_size=14, legend_size=13)
    
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    legend = ax.legend(loc='best', fontsize=13, framealpha=0.95)
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontfamily('serif')
        text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "ours_prediction_error_over_time.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[信息] 保存图像: {output_file}")


def plot_prediction_control_correlation(ours_data: pd.DataFrame, output_dir: str):
    """绘制平台预测误差与机器狗基座和平台的控制误差相关性图像。
    
    Args:
        ours_data: Ours方法的DataFrame
        output_dir: 输出目录
    """
    if ours_data is None:
        print("[警告] Ours方法数据为空，跳过此图")
        return
    
    if 'prediction_error' not in ours_data.columns or 'control_error' not in ours_data.columns:
        print("[警告] Ours方法数据中缺少预测误差或控制误差数据，跳过此图")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # 获取有效数据
    valid_mask = (ours_data['prediction_error'].notna() & 
                 ours_data['control_error'].notna())
    pred_error = ours_data.loc[valid_mask, 'prediction_error']
    control_error = ours_data.loc[valid_mask, 'control_error']
    
    if len(pred_error) < 2:
        print("[警告] 有效数据点不足，无法绘制相关性图")
        return
    
    # 使用Ours方法的配置
    ours_config = METHOD_CONFIGS.get('Ours', {})
    ours_color = ours_config.get('color', '#E91E63')
    ours_alpha = ours_config.get('alpha', 1.0)
    
    # 散点图（使用Ours的颜色，但稍微透明以便看到重叠点）
    ax.scatter(control_error, pred_error, alpha=0.7, s=40, 
              color=ours_color, edgecolors='white', linewidths=0.5, zorder=10)
    
    # 计算相关系数
    if len(pred_error) > 1:
        pred_std = pred_error.std()
        control_std = control_error.std()
        
        if pred_std > 1e-10 and control_std > 1e-10:
            try:
                corr, p_value = pearsonr(pred_error, control_error)
                
                # 添加趋势线
                try:
                    z = np.polyfit(control_error, pred_error, 1)
                    p = np.poly1d(z)
                    x_fit = np.linspace(control_error.min(), control_error.max(), 100)
                    y_fit = p(x_fit)
                    ax.plot(x_fit, y_fit, "r--", alpha=1.0, linewidth=2.5, 
                           label=f'Linear Fit (r={corr:.4f}, p={p_value:.4f})')
                except Exception:
                    # 如果拟合失败，只显示相关系数
                    ax.text(0.05, 0.95, f'Correlation: r={corr:.4f}, p={p_value:.4f}', 
                           transform=ax.transAxes, fontsize=14, verticalalignment='top',
                           fontfamily='serif', fontname='Times New Roman',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except Exception as e:
                print(f"[警告] 计算相关系数失败: {e}")
    
    ax.set_xlabel('Control Error (rad)', fontsize=16, fontweight='bold', fontfamily='serif')
    ax.set_ylabel('Prediction Error (rad)', fontsize=16, fontweight='bold', fontfamily='serif')
    ax.set_title('Correlation: Prediction Error vs Control Error (Ours Method)', 
                fontsize=18, fontweight='bold', pad=15, fontfamily='serif')
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    setup_plot_fonts(ax, title_size=18, label_size=16, tick_size=14, legend_size=13)
    if ax.get_legend_handles_labels()[0]:
        legend = ax.legend(loc='best', fontsize=13, framealpha=0.95)
        # 设置图例字体
        for text in legend.get_texts():
            text.set_fontfamily('serif')
            text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "ours_prediction_control_correlation.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[信息] 保存图像: {output_file}")


def save_metrics_table(method_metrics: dict, output_dir: str):
    """保存指标对比表格。
    
    Args:
        method_metrics: 字典，键为方法名，值为指标字典
        output_dir: 输出目录
    """
    # 准备数据
    methods = list(method_metrics.keys())
    data = {
        'Method': methods,
        'Control Error RMSE (rad)': [method_metrics[m].get('control_rmse', 0.0) for m in methods],
        'Survival Rate': [method_metrics[m].get('survival_rate', 0.0) for m in methods],
        'Average Energy Consumption': [method_metrics[m].get('avg_energy', 0.0) for m in methods],
    }
    
    df = pd.DataFrame(data)
    
    # 保存为CSV
    csv_file = os.path.join(output_dir, "comparison_metrics.csv")
    df.to_csv(csv_file, index=False)
    print(f"[信息] 保存指标表格: {csv_file}")
    
    # 保存为LaTeX表格
    latex_file = os.path.join(output_dir, "comparison_metrics.tex")
    with open(latex_file, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))
    print(f"[信息] 保存LaTeX表格: {latex_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare different methods for platform following task")
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs='+',
        required=True,
        help="List of data directories for different methods. Format: method_name:path (e.g., 'Ours:/path/to/ours', 'React-PPO:/path/to/react_ppo')"
    )
    parser.add_argument(
        "--max_episode_length",
        type=float,
        default=1000.0,
        help="Maximum episode length for computing survival rate (default: 1000.0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/user/IsaacLab/comparison_results",
        help="Output directory for comparison results"
    )
    parser.add_argument(
        "--data_transform",
        type=str,
        default=None,
        help="JSON string for data transformation. Format: {\"method_name\": {\"column_name\": {\"operation\": \"multiply\", \"value\": 1000}}}. Example: '{\"Ours\": {\"control_error\": {\"operation\": \"multiply\", \"value\": 1000}}}'"
    )
    parser.add_argument(
        "--plot_config",
        type=str,
        default=None,
        help="JSON string for plot data combination. Format: {\"plot_name\": {\"method_name\": {\"expression\": \"column1 * 0.5 + column2\"}}}. Example: '{\"base_error_ratio_mse\": {\"Ours\": {\"expression\": \"base_error_ratio * 0.5 + control_error\"}}}'"
    )
    
    args = parser.parse_args()
    
    # 解析数据变换配置
    data_transforms = {}
    if args.data_transform:
        try:
            data_transforms = json.loads(args.data_transform)
            print(f"[信息] 数据变换配置: {data_transforms}")
        except json.JSONDecodeError as e:
            print(f"[警告] 数据变换配置JSON解析失败: {e}，将忽略变换")
            data_transforms = {}
    
    # 解析绘图配置
    plot_configs = {}
    if args.plot_config:
        try:
            plot_configs = json.loads(args.plot_config)
            print(f"[信息] 绘图配置: {plot_configs}")
        except json.JSONDecodeError as e:
            print(f"[警告] 绘图配置JSON解析失败: {e}，将忽略配置")
            plot_configs = {}
    
    # 创建输出目录（使用绝对路径）
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        print(f"[错误] 无法创建输出目录: {output_dir}")
        print(f"[错误] 请检查路径权限或使用其他目录")
        return
    except Exception as e:
        print(f"[错误] 创建输出目录失败: {e}")
        return
    
    # 解析方法数据目录
    method_data = {}
    method_metrics = {}
    method_stats = {}
    
    print("=" * 80)
    print("开始加载各方法的数据...")
    print("=" * 80)
    
    for method_dir_str in args.data_dirs:
        if ':' not in method_dir_str:
            print(f"[警告] 跳过无效的格式: {method_dir_str} (应为 method_name:path)")
            continue
        
        method_name, data_dir = method_dir_str.split(':', 1)
        method_name = method_name.strip()
        data_dir = os.path.abspath(os.path.expanduser(data_dir.strip()))
        
        if not os.path.exists(data_dir):
            print(f"[警告] 数据目录不存在: {data_dir}，跳过方法 {method_name}")
            continue
        
        print(f"\n[信息] 加载方法: {method_name}")
        print(f"[信息] 数据目录: {data_dir}")
        
        # 获取该方法的数据变换配置（如果有）
        method_transform = data_transforms.get(method_name, None)
        
        # 加载时间序列数据
        df = load_csv_data(data_dir, data_transform=method_transform)
        method_data[method_name] = df
        
        # 加载统计数据
        stats = load_statistics(data_dir)
        method_stats[method_name] = stats
        
        # 计算指标
        if df is not None:
            # 从统计数据中获取最大episode长度（如果存在），否则使用命令行参数
            max_episode_length = stats.get('max_episode_length', args.max_episode_length)
            metrics = compute_metrics(df, stats, max_episode_length)
            method_metrics[method_name] = metrics
        else:
            method_metrics[method_name] = {}
    
    print("\n" + "=" * 80)
    print("开始生成对比图表...")
    print("=" * 80)
    
    # 获取各图表的绘图配置
    base_error_config = plot_configs.get('base_error_ratio_mse', {})
    roll_angle_config = plot_configs.get('roll_angle', {})
    
    # 1. 各方法基座误差比值的MSE随时间步的变化
    plot_base_error_ratio_mse_comparison(method_data, output_dir, base_error_config)
    
    # 2. 各方法机器狗横滚角和平台横滚角随时间变化曲线对比
    plot_roll_angle_comparison(method_data, output_dir, roll_angle_config)
    
    # 3. 各方法的指标对比（柱状图）
    plot_metrics_comparison(method_metrics, output_dir)
    
    # 4. Ours方法的预测误差随时间变化
    if 'Ours' in method_data:
        plot_prediction_error_vs_steps(method_data['Ours'], output_dir)
    
    # 5. Ours方法的预测误差与控制误差相关性
    if 'Ours' in method_data:
        plot_prediction_control_correlation(method_data['Ours'], output_dir)
    
    # 6. 保存指标表格
    save_metrics_table(method_metrics, output_dir)
    
    print("\n" + "=" * 80)
    print("对比实验完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

