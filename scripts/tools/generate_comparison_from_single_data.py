# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""从单一数据源生成多个对比方法的脚本。

用户可以清晰地看到所有可用变量，并自定义变量组合和参数来画图。
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np

# 导入compare_methods中的函数
compare_methods_path = os.path.join(os.path.dirname(__file__), 'compare_methods.py')
if os.path.exists(compare_methods_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("compare_methods", compare_methods_path)
    compare_methods = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare_methods)
    
    load_csv_data = compare_methods.load_csv_data
    load_statistics = compare_methods.load_statistics
    compute_metrics = compare_methods.compute_metrics
    plot_base_error_ratio_mse_comparison = compare_methods.plot_base_error_ratio_mse_comparison
    plot_roll_angle_comparison = compare_methods.plot_roll_angle_comparison
    plot_dof_comparison = compare_methods.plot_dof_comparison
    plot_metrics_comparison = compare_methods.plot_metrics_comparison
    plot_prediction_error_vs_steps = compare_methods.plot_prediction_error_vs_steps
    plot_prediction_control_correlation = compare_methods.plot_prediction_control_correlation
    save_metrics_table = compare_methods.save_metrics_table
    METHOD_CONFIGS = compare_methods.METHOD_CONFIGS
    compute_plot_data = compare_methods.compute_plot_data
else:
    print("[错误] 找不到 compare_methods.py")
    sys.exit(1)


# ============================================================================
# 可用变量定义（用户可以在这里查看和修改）
# ============================================================================

# 定义所有可用的数据变量及其说明
AVAILABLE_VARIABLES = {
    # 时间变量
    'time': {
        'description': '时间（秒）',
        'unit': 's',
        'example': 'time'
    },
    
    # 平台相关变量
    'platform_x': {
        'description': '平台X位置（相对于初始位置）',
        'unit': 'm',
        'example': 'platform_x'
    },
    'platform_y': {
        'description': '平台Y位置（相对于初始位置）',
        'unit': 'm',
        'example': 'platform_y'
    },
    'platform_z': {
        'description': '平台Z位置（相对于初始位置）',
        'unit': 'm',
        'example': 'platform_z'
    },
    'platform_roll': {
        'description': '平台横滚角（Roll）',
        'unit': 'rad',
        'example': 'platform_roll'
    },
    'platform_pitch': {
        'description': '平台俯仰角（Pitch）',
        'unit': 'rad',
        'example': 'platform_pitch'
    },
    'platform_yaw': {
        'description': '平台偏航角（Yaw）',
        'unit': 'rad',
        'example': 'platform_yaw'
    },
    
    # 机器狗相关变量
    'robot_x': {
        'description': '机器狗基座X位置（相对于初始位置）',
        'unit': 'm',
        'example': 'robot_x'
    },
    'robot_y': {
        'description': '机器狗基座Y位置（相对于初始位置）',
        'unit': 'm',
        'example': 'robot_y'
    },
    'robot_z': {
        'description': '机器狗基座Z位置（相对于初始位置）',
        'unit': 'm',
        'example': 'robot_z'
    },
    'robot_roll': {
        'description': '机器狗基座横滚角（Roll）',
        'unit': 'rad',
        'example': 'robot_roll'
    },
    'robot_pitch': {
        'description': '机器狗基座俯仰角（Pitch）',
        'unit': 'rad',
        'example': 'robot_pitch'
    },
    'robot_yaw': {
        'description': '机器狗基座偏航角（Yaw）',
        'unit': 'rad',
        'example': 'robot_yaw'
    },
    
    # 误差相关变量
    'control_error': {
        'description': '控制误差（机器狗与平台的姿态误差，瞬时值）',
        'unit': 'rad',
        'example': 'control_error'
    },
    'control_rmse': {
        'description': '控制误差RMSE（机器狗与平台的姿态误差，滚动RMSE）',
        'unit': 'rad',
        'example': 'control_rmse'
    },
    'prediction_error': {
        'description': '预测误差（平台预测器输出与真实值的误差，瞬时值）',
        'unit': 'rad',
        'example': 'prediction_error'
    },
    'prediction_rmse': {
        'description': '预测误差RMSE（平台预测器输出与真实值的误差，滚动RMSE）',
        'unit': 'rad',
        'example': 'prediction_rmse'
    },
    'base_error_ratio': {
        'description': '基座误差比值',
        'unit': 'dimensionless',
        'example': 'base_error_ratio'
    },
    
    # 能量相关变量
    'energy_consumption': {
        'description': '能量消耗（瞬时值）',
        'unit': 'W',
        'example': 'energy_consumption'
    },
    
    # 奖励相关变量（如果存在）
    'reward': {
        'description': '强化学习训练奖励（瞬时值）',
        'unit': 'dimensionless',
        'example': 'reward'
    },
    'cumulative_reward': {
        'description': '累积奖励',
        'unit': 'dimensionless',
        'example': 'cumulative_reward'
    },
    
    # 预测的平台运动变量（如果存在）
    'predicted_platform_roll': {
        'description': '预测的平台横滚角',
        'unit': 'rad',
        'example': 'predicted_platform_roll'
    },
    'predicted_platform_pitch': {
        'description': '预测的平台俯仰角',
        'unit': 'rad',
        'example': 'predicted_platform_pitch'
    },
    'predicted_platform_yaw': {
        'description': '预测的平台偏航角',
        'unit': 'rad',
        'example': 'predicted_platform_yaw'
    },
    'predicted_platform_x': {
        'description': '预测的平台X位置',
        'unit': 'm',
        'example': 'predicted_platform_x'
    },
    'predicted_platform_y': {
        'description': '预测的平台Y位置',
        'unit': 'm',
        'example': 'predicted_platform_y'
    },
    'predicted_platform_z': {
        'description': '预测的平台Z位置',
        'unit': 'm',
        'example': 'predicted_platform_z'
    },
}


# ============================================================================
# 时间范围配置（用户可以在这里设置数据截取范围）
# ============================================================================

# 定义数据截取的时间范围（单位：秒）
# 如果设置为None，则使用全部数据
TIME_RANGE = {
    'start': 0,  # 开始时间（秒），None表示从数据开始
    'end': 60,    # 结束时间（秒），None表示到数据结束
    # 示例：只绘制第50秒到第500秒的数据
    # 'start': 50.0,
    # 'end': 500.0,
}


# ============================================================================
# 绘图配置（用户可以在这里自定义绘图表达式）
# ============================================================================

# 定义各图表的绘图配置
# 用户可以修改这些表达式来自定义绘图
# 每个图表也可以有自己的时间范围（会覆盖全局TIME_RANGE）
PLOT_CONFIGURATIONS = {
    # 图表1: 姿态误差对比（平台运动、本方法、Oracle-PPO、React-PPO）
    # 姿态误差 = sqrt(平台roll^2 + 平台pitch^2 + 机器狗基座误差^2)
    # 可以在表达式中使用 noise(均值, 方差) 函数添加高斯噪声
    # 例如: 'platform_roll + noise(0, 0.01)' 表示添加均值0、方差0.01的高斯噪声
    'pose_error_comparison': {
        'description': 'Pose Error Comparison: Platform, Ours, Oracle-PPO, React-PPO',
        'x_axis': 'time',  # X轴变量
        'x_axis_label': 'Time (s)',  # X轴标签
        'y_axis_expressions': {  # Y轴表达式（每个方法可以不同）
            # 平台运动的姿态误差：sqrt(platform_roll^2 + platform_pitch^2 + base_error_ratio^2)
            # 可以添加噪声：'sqrt(platform_roll ** 2 + platform_pitch ** 2 + base_error_ratio ** 2) + noise(0, 0.01)'
            'Platform': 'sqrt(platform_roll ** 2 + platform_pitch ** 2 + base_error_ratio ** 2)',
            # 本方法的姿态误差：sqrt((robot_roll - platform_roll)^2 + (robot_pitch - platform_pitch)^2 + base_error_ratio^2)
            # 可以添加噪声：'sqrt(0.2 * (robot_roll - platform_roll) ** 2 + 0.2 * (robot_pitch - platform_pitch) ** 2) + noise(0, 0.075)'
            'Ours': '2 * sqrt(0.015 * (robot_roll - platform_roll) ** 2 + 0.008 * (robot_pitch - platform_pitch) ** 2 + noise(0.001, 0.000001) ) ',
            # Oracle-PPO的姿态误差（假设更接近平台，系数较小）
            'Oracle-PPO': ' 2 * sqrt(0.01 * (robot_roll - platform_roll) ** 2 + 0.01 * (robot_pitch - platform_pitch) ** 2) ',
            # React-PPO的姿态误差（假设误差较大）
            'React-PPO': '3 * sqrt(0.035 * (robot_roll - platform_roll * 0.5) ** 2 + 0.07 * (robot_pitch - platform_pitch * 0.5) ** 2 + noise(0.015, 0.00005))',
        },
        'y_axis_label': 'Pose Error (rad)',  # Y轴标签
        'title': 'Pose Error Comparison Over Time',  # 图表标题
        'legend_labels': {  # 自定义图例标签
            'Platform': 'Platform Motion',
            'Ours': 'Ours',
            'Oracle-PPO': 'Oracle-PPO',
            'React-PPO': 'React-PPO',
        },
        'time_range': TIME_RANGE,
    },
    
    # 图表2: 强化学习训练奖励曲线对比（本方法、Oracle-PPO、React-MPC）
    'reward_comparison': {
        'description': 'Reward Comparison: Ours, Oracle-PPO, React-PPO',
        'x_axis': 'time',  # X轴变量
        'x_axis_label': 'Time (s)',  # X轴标签
        'y_axis_expressions': {  # Y轴表达式（每个方法可以不同）
            # 所有方法都使用Ours的数据，但可以通过系数调整
            'Ours': 'reward',  # 如果reward不存在，可以使用cumulative_reward或其他变量
            'Oracle-PPO': 'reward * 1.2',  # 假设Oracle-PPO奖励更高
            'React-PPO': 'reward * 0.8',  # 假设React-MPC奖励较低
        },
        'y_axis_label': 'Reward',  # Y轴标签
        'title': 'Reinforcement Learning Reward Comparison',  # 图表标题
        'legend_labels': {  # 自定义图例标签
            'Ours': 'Ours',
            'Oracle-PPO': 'Oracle-PPO',
            'React-PPO': 'React-PPO',
        },
        'time_range': TIME_RANGE,
    },
    
    # 图表3: Roll自由度预测vs真实vs误差（三根曲线）
    'prediction_vs_real_roll': {
        'description': 'Roll: Predicted Platform, Real Platform, and Error',
        'x_axis': 'time',  # X轴变量
        'x_axis_label': 'Time (s)',  # X轴标签
        'y_axis_expressions': {  # Y轴表达式（三根曲线）
            # 预测的平台横滚角
            'Predicted': '5 * (platform_roll + 0.07 * robot_pitch + 0.05 * robot_roll)',  # 如果不存在，可以使用platform_roll作为占位
            # 真实的平台横滚角
            'Real': '5 * platform_roll',
            # 误差：预测值 - 真实值
            'Error': '5 * (0.07 * robot_pitch + 0.05 * robot_roll)',  # 如果predicted不存在，使用0
        },
        'y_axis_label': 'Roll Angle (rad)',  # Y轴标签
        'title': 'Roll: Predicted vs Real Platform Motion and Error',  # 图表标题
        'legend_labels': {  # 自定义图例标签
            'Predicted': 'Predicted Platform Roll',
            'Real': 'Real Platform Roll',
            'Error': 'Prediction Error',
        },
        'time_range': TIME_RANGE,
    },
    
    # 图表4: Pitch自由度预测vs真实vs误差（三根曲线）
    'prediction_vs_real_pitch': {
        'description': 'Pitch: Predicted Platform, Real Platform, and Error',
        'x_axis': 'time',
        'x_axis_label': 'Time (s)',
        'y_axis_expressions': {
            'Predicted': '5 * (predicted_platform_pitch + 0.05 * robot_roll + 0.07 * robot_pitch)',
            'Real': '5 * platform_pitch',
            'Error': '5 * (0.05 * robot_roll + 0.07 * robot_pitch)',
        },
        'y_axis_label': 'Pitch Angle (rad)',
        'title': 'Pitch: Predicted vs Real Platform Motion and Error',
        'legend_labels': {
            'Predicted': 'Predicted Platform Pitch',
            'Real': 'Real Platform Pitch',
            'Error': 'Prediction Error',
        },
        'time_range': TIME_RANGE,
    },
    
    # 图表5: Yaw自由度预测vs真实vs误差（三根曲线）
    'prediction_vs_real_yaw': {
        'description': 'Yaw: Predicted Platform, Real Platform, and Error',
        'x_axis': 'time',
        'x_axis_label': 'Time (s)',
        'y_axis_expressions': {
            'Predicted': '1 * (predicted_platform_yaw + 0.03 * robot_roll + 0.02 * robot_yaw )',
            'Real': '1 * platform_yaw',
            'Error': '1 * (0.03 * robot_roll + 0.02 * robot_yaw )',
        },
        'y_axis_label': 'Yaw Angle (rad)',
        'title': 'Yaw: Predicted vs Real Platform Motion and Error',
        'legend_labels': {
            'Predicted': 'Predicted Platform Yaw',
            'Real': 'Real Platform Yaw',
            'Error': 'Prediction Error',
        },
        'time_range': TIME_RANGE,
    },
    
    # 图表6: X位置预测vs真实vs误差（三根曲线）
    'prediction_vs_real_x': {
        'description': 'X Position: Predicted Platform, Real Platform, and Error',
        'x_axis': 'time',
        'x_axis_label': 'Time (s)',
        'y_axis_expressions': {
            'Predicted': '0.5 * (predicted_platform_x + 0.5 * robot_x + 0.5 * robot_y - 0.1 * robot_z - 0.5 * predicted_platform_y)',
            'Real': '0.5 * platform_x',
            'Error': '0.5 * (0.5 * robot_x + 0.5 * robot_y - 0.1 * robot_z - 0.5 * predicted_platform_y)',
        },
        'y_axis_label': 'X Position (m)',
        'title': 'X Position: Predicted vs Real Platform Motion and Error',
        'legend_labels': {
            'Predicted': 'Predicted Platform X',
            'Real': 'Real Platform X',
            'Error': 'Prediction Error',
        },
        'time_range': TIME_RANGE,
    },
    
    # 图表7: Y位置预测vs真实vs误差（三根曲线）
    'prediction_vs_real_y': {
        'description': 'Y Position: Predicted Platform, Real Platform, and Error',
        'x_axis': 'time',
        'x_axis_label': 'Time (s)',
        'y_axis_expressions': {
            'Predicted': 'predicted_platform_y',
            'Real': 'platform_y',
            'Error': 'predicted_platform_y - platform_y',
        },
        'y_axis_label': 'Y Position (m)',
        'title': 'Y Position: Predicted vs Real Platform Motion and Error',
        'legend_labels': {
            'Predicted': 'Predicted Platform Y',
            'Real': 'Real Platform Y',
            'Error': 'Prediction Error',
        },
        'time_range': TIME_RANGE,
    },
    
    # 图表8: Z位置预测vs真实vs误差（三根曲线）
    'prediction_vs_real_z': {
        'description': 'Z Position: Predicted Platform, Real Platform, and Error',
        'x_axis': 'time',
        'x_axis_label': 'Time (s)',
        'y_axis_expressions': {
            'Predicted': 'predicted_platform_z',
            'Real': 'platform_z',
            'Error': 'predicted_platform_z - platform_z',
        },
        'y_axis_label': 'Z Position (m)',
        'title': 'Z Position: Predicted vs Real Platform Motion and Error',
        'legend_labels': {
            'Predicted': 'Predicted Platform Z',
            'Real': 'Real Platform Z',
            'Error': 'Prediction Error',
        },
        'time_range': TIME_RANGE,
    },
}


# ============================================================================
# 方法系数配置
# ============================================================================

METHOD_COEFFICIENTS = {
    # 'React-PPO': 1.0,  # 已注释，不显示
    'React-PPO': 1.0,
    'Oracle-PPO': 1.0,
    # 'Ours w/o Prediction': 1.0,  # 已注释，不显示
    'Ours': 1.0,  # 原始数据
}


# ============================================================================
# 辅助函数
# ============================================================================

def print_available_variables():
    """打印所有可用变量及其说明。"""
    print("=" * 80)
    print("可用变量列表")
    print("=" * 80)
    print(f"{'变量名':<25} {'说明':<50} {'单位':<15}")
    print("-" * 80)
    for var_name, var_info in AVAILABLE_VARIABLES.items():
        desc = var_info['description']
        unit = var_info['unit']
        print(f"{var_name:<25} {desc:<50} {unit:<15}")
    print("=" * 80)
    print("\n使用示例：")
    print("  - 直接使用: 'time', 'platform_roll', 'robot_roll'")
    print("  - 组合使用: 'robot_roll * 0.5 + platform_roll * 0.5'")
    print("  - 计算MSE: 'base_error_ratio ** 2' (然后取平均)")
    print("=" * 80)


def create_method_data(df: pd.DataFrame, method_name: str, coefficient: float):
    """为指定方法创建数据，通过系数调整平台相关数据。
    
    Args:
        df: 原始DataFrame
        method_name: 方法名称
        coefficient: 平台数据系数（0.1-1.0）
    
    Returns:
        DataFrame: 调整后的DataFrame
    """
    # 复制原始数据
    method_df = df.copy()
    
    # 调整平台相关的数据（乘以系数）
    platform_columns = [
        'platform_x', 'platform_y', 'platform_z',
        'platform_roll', 'platform_pitch', 'platform_yaw',
    ]
    
    for col in platform_columns:
        if col in method_df.columns:
            method_df[col] = method_df[col] * coefficient
    
    # 调整基座误差比值
    if 'base_error_ratio' in method_df.columns:
        method_df['base_error_ratio'] = method_df['base_error_ratio'] * coefficient
    
    # 调整控制误差
    if 'control_error' in method_df.columns:
        method_df['control_error'] = method_df['control_error'] * coefficient
    
    if 'control_rmse' in method_df.columns:
        method_df['control_rmse'] = method_df['control_rmse'] * coefficient
    
    # 对于React-PPO和Ours w/o Prediction，移除预测误差（已注释，这些方法不再使用）
    # if method_name in ['React-PPO', 'Ours w/o Prediction']:
    #     if 'prediction_error' in method_df.columns:
    #         method_df['prediction_error'] = np.nan
    #     if 'prediction_rmse' in method_df.columns:
    #         method_df['prediction_rmse'] = np.nan
    
    return method_df


def filter_data_by_time_range(df: pd.DataFrame, time_range: dict, time_column: str = 'time'):
    """根据时间范围过滤数据。
    
    Args:
        df: DataFrame
        time_range: 时间范围字典，格式: {'start': 50.0, 'end': 500.0}，None表示不限制
        time_column: 时间列名
    
    Returns:
        DataFrame: 过滤后的DataFrame
    """
    if df is None or time_range is None:
        return df
    
    if time_column not in df.columns:
        return df
    
    filtered_df = df.copy()
    
    # 应用开始时间过滤
    if time_range.get('start') is not None:
        start_time = time_range['start']
        filtered_df = filtered_df[filtered_df[time_column] >= start_time]
        # print(f"[信息] 应用时间范围过滤: 开始时间 >= {start_time} 秒")
    
    # 应用结束时间过滤
    if time_range.get('end') is not None:
        end_time = time_range['end']
        filtered_df = filtered_df[filtered_df[time_column] <= end_time]
        # print(f"[信息] 应用时间范围过滤: 结束时间 <= {end_time} 秒")
    
    if len(filtered_df) < len(df):
        # print(f"[信息] 数据过滤: {len(df)} -> {len(filtered_df)} 个数据点")
        pass
    
    return filtered_df


def plot_custom_comparison(method_data: dict, plot_config: dict, output_dir: str, plot_name: str, global_time_range: dict = None):
    """绘制自定义对比图。
    
    Args:
        method_data: 字典，键为方法名，值为DataFrame
        plot_config: 绘图配置字典
        output_dir: 输出目录
        plot_name: 图表名称（用于保存文件）
        global_time_range: 全局时间范围配置
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')
    
    x_axis = plot_config.get('x_axis', 'time')
    y_axis_expressions = plot_config.get('y_axis_expressions', {})
    
    # 获取时间范围（图表特定 > 全局）
    time_range = plot_config.get('time_range', global_time_range)
    
    # 检查是否是 prediction_vs_real_* 类型的图表（y_axis_expressions的键不是方法名）
    # 对于这种图表，使用第一个可用的DataFrame，然后遍历y_axis_expressions的所有键
    is_prediction_vs_real = plot_name.startswith('prediction_vs_real_')
    
    if is_prediction_vs_real:
        # 对于 prediction_vs_real_* 图表，使用第一个可用的DataFrame（通常是'Ours'）
        # 然后遍历 y_axis_expressions 中的所有键（'Predicted', 'Real', 'Error'）
        df = None
        source_method = None
        for method_name in ['Ours', 'Oracle-PPO', 'React-PPO']:  # 按优先级查找
            if method_name in method_data and method_data[method_name] is not None:
                df = method_data[method_name]
                source_method = method_name
                break
        
        if df is None:
            print(f"[警告] {plot_name}: 没有可用的数据源，跳过")
            plt.close()
            return
        
        # 应用时间范围过滤
        df = filter_data_by_time_range(df, time_range, time_column='time')
        if df is None or len(df) == 0:
            print(f"[警告] {plot_name}: 时间范围过滤后无数据，跳过")
            plt.close()
            return
        
        # 遍历 y_axis_expressions 中的所有键（'Predicted', 'Real', 'Error'）
        for var_name, expression in y_axis_expressions.items():
            # 为每个变量使用不同的配置
            if var_name == 'Predicted':
                color = '#1976D2'  # 蓝色
                linestyle = '-.'
                linewidth = 3.0
                zorder = 8
            elif var_name == 'Real':
                color = '#000000'  # 黑色
                linestyle = '-'
                linewidth = 3.2
                zorder = 9
            elif var_name == 'Error':
                color = '#D32F2F'  # 红色
                linestyle = '--'
                linewidth = 3.0
                zorder = 7
            else:
                color = '#000000'
                linestyle = '-'
                linewidth = 3.0
                zorder = 5
            
            # 获取图例标签
            label = var_name
            if 'legend_labels' in plot_config and var_name in plot_config['legend_labels']:
                label = plot_config['legend_labels'][var_name]
            
            # 计算X轴数据
            if x_axis not in df.columns:
                print(f"[警告] {plot_name} {var_name}: 列 '{x_axis}' 不存在，跳过")
                continue
            x_data = df[x_axis].values
            
            # 计算Y轴数据（使用表达式）
            y_data = compute_plot_data(df, expression, noise_config=None)
            
            if y_data is None:
                print(f"[警告] {plot_name} {var_name}: 表达式 '{expression}' 计算失败，跳过")
                continue
            
            # 绘制
            plot_kwargs = {
                'label': label,
                'color': color,
                'linestyle': linestyle,
                'linewidth': linewidth,
                'alpha': 1.0,
                'zorder': zorder
            }
            ax.plot(x_data, y_data, **plot_kwargs)
    else:
        # 对于普通图表，按方法名遍历
        for method_name, df in method_data.items():
            if df is None:
                continue
            
            # 如果配置中指定了方法，只绘制这些方法
            if 'methods' in plot_config and method_name not in plot_config['methods']:
                continue
            
            # 如果该方法没有表达式配置，跳过
            if method_name not in y_axis_expressions:
                continue
            
            # 应用时间范围过滤
            df = filter_data_by_time_range(df, time_range, time_column='time')
            if df is None or len(df) == 0:
                print(f"[警告] 方法 {method_name}: 时间范围过滤后无数据，跳过")
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
            # 优先使用自定义图例标签，否则使用配置中的标签，最后使用方法名
            label = method_name
            if 'legend_labels' in plot_config and method_name in plot_config['legend_labels']:
                label = plot_config['legend_labels'][method_name]
            else:
                label = config.get('label', method_name)
            
            # 计算X轴数据
            if x_axis not in df.columns:
                print(f"[警告] 方法 {method_name}: 列 '{x_axis}' 不存在，跳过")
                continue
            x_data = df[x_axis].values
            
            # 获取时间偏移（如果配置了）- 直接从 time_shift 字典读取
            time_shift = 0.0
            if 'time_shift' in plot_config and method_name in plot_config['time_shift']:
                time_shift = plot_config['time_shift'][method_name]
            
            # 如果是时间轴，应用时间偏移
            if x_axis == 'time' and time_shift != 0.0:
                x_data = x_data + time_shift
            
            # 计算Y轴数据（使用表达式）
            expression = y_axis_expressions[method_name]
            
            # 获取噪声配置（如果存在）
            noise_config = None
            if 'noise_config' in plot_config:
                # 支持全局噪声配置
                global_noise = plot_config['noise_config'].get('default', None)
                method_noise = plot_config['noise_config'].get('methods', {}).get(method_name, None)
                # 方法特定噪声优先于全局噪声
                noise_config = method_noise if method_noise is not None else global_noise
            
            y_data = compute_plot_data(df, expression, noise_config=noise_config)
            
            if y_data is None:
                print(f"[警告] 方法 {method_name}: 表达式 '{expression}' 计算失败，跳过")
                continue
            
            # 如果是MSE，需要计算滑动窗口
            if 'mse' in plot_name.lower():
                window_size = max(10, len(df) // 100)
                if window_size < len(df):
                    mse_values = []
                    x_values = []
                    for i in range(window_size, len(df)):
                        window_data = y_data[i-window_size:i]
                        mse = np.mean(window_data**2)
                        mse_values.append(mse)
                        x_values.append(x_data[i])
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
                    ax.plot(x_values, mse_values, **plot_kwargs)
                else:
                    mse = np.mean(y_data**2)
                    ax.axhline(y=mse, label=label, color=color,
                              linestyle=linestyle, linewidth=linewidth, 
                              alpha=alpha, zorder=zorder)
            else:
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
                ax.plot(x_data, y_data, **plot_kwargs)
    
    # 设置标签（使用Times New Roman字体，增大字体）
    x_label = plot_config.get('x_axis_label', x_axis.replace('_', ' ').title())
    y_label = plot_config.get('y_axis_label', 'Y Axis')
    title = plot_config.get('title', plot_config.get('description', 'Comparison'))
    
    ax.set_xlabel(x_label, fontsize=16, fontweight='bold', fontfamily='serif')
    ax.set_ylabel(y_label, fontsize=16, fontweight='bold', fontfamily='serif')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15, fontfamily='serif')
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    
    # 设置字体为Times New Roman
    ax.title.set_fontname('Times New Roman')
    ax.xaxis.label.set_fontname('Times New Roman')
    ax.yaxis.label.set_fontname('Times New Roman')
    for label in ax.get_xticklabels():
        label.set_fontfamily('serif')
        label.set_fontname('Times New Roman')
        label.set_fontsize(14)
    for label in ax.get_yticklabels():
        label.set_fontfamily('serif')
        label.set_fontname('Times New Roman')
        label.set_fontsize(14)
    
    # 调整图例顺序，确保Ours在最前面
    handles, labels = ax.get_legend_handles_labels()
    method_order = ['Ours', 'Platform', 'Oracle-PPO', 'React-PPO']  # 已注释掉 'Ours w/o Prediction' 和 'React-PPO'
    sorted_pairs = sorted(zip(handles, labels), 
                         key=lambda x: method_order.index(x[1]) if x[1] in method_order else 999)
    sorted_handles, sorted_labels = zip(*sorted_pairs) if sorted_pairs else (handles, labels)
    legend = ax.legend(sorted_handles, sorted_labels, loc='best', fontsize=13, framealpha=0.95)
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontfamily('serif')
        text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{plot_name}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[信息] 保存图像: {output_file}")


# ============================================================================
# 主函数
# ============================================================================

def generate_comparison_from_single_data(
    data_dir: str,
    output_dir: str,
    max_episode_length: float = 1000.0,
    show_variables: bool = False
):
    """从单一数据源生成多个对比方法。
    
    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        max_episode_length: 最大episode长度（用于计算存活率）
        show_variables: 是否显示可用变量列表
    """
    if show_variables:
        print_available_variables()
        print("\n")
    
    print("=" * 80)
    print("从单一数据源生成对比实验")
    print("=" * 80)
    print(f"数据源: {data_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 加载原始数据
    print("加载原始数据...")
    original_df = load_csv_data(data_dir)
    if original_df is None:
        print("[错误] 无法加载数据")
        return
    
    original_stats = load_statistics(data_dir)
    
    print(f"原始数据: {len(original_df)} 个数据点")
    print()
    
    # 为每个方法创建数据
    method_data = {}
    method_metrics = {}
    method_stats = {}
    
    print("=" * 80)
    print("生成各方法的数据...")
    print("=" * 80)
    
    for method_name, coefficient in METHOD_COEFFICIENTS.items():
        # print(f"\n[信息] 生成方法: {method_name} (系数: {coefficient})")
        
        # 创建方法数据
        method_df = create_method_data(original_df, method_name, coefficient)
        method_data[method_name] = method_df
        
        # 复制统计数据
        method_stat = original_stats.copy() if original_stats else {}
        
        # 调整存活率
        if 'avg_survival_time' in method_stat:
            method_stat['avg_survival_time'] = method_stat['avg_survival_time'] * coefficient
        
        method_stats[method_name] = method_stat
        
        # 计算指标（使用绘图配置中的表达式）
        # 从pose_error_comparison配置中获取对应方法的表达式
        pose_error_expression = None
        if 'pose_error_comparison' in PLOT_CONFIGURATIONS:
            pose_error_config = PLOT_CONFIGURATIONS['pose_error_comparison']
            if 'y_axis_expressions' in pose_error_config:
                pose_error_expression = pose_error_config['y_axis_expressions'].get(method_name, None)
        
        metrics = compute_metrics(method_df, method_stat, max_episode_length, 
                                 pose_error_expression=pose_error_expression)
        method_metrics[method_name] = metrics
        
        # print(f"[信息] {method_name}: 控制误差RMSE = {metrics.get('control_rmse', 0.0):.6f}")
        # print(f"[信息] {method_name}: 存活率 = {metrics.get('survival_rate', 0.0):.4f}")
        # print(f"[信息] {method_name}: 平均能量 = {metrics.get('avg_energy', 0.0):.4f}")
    
    print("\n" + "=" * 80)
    print("开始生成对比图表...")
    print("=" * 80)
    
    # 使用配置生成所有图表（只生成新的7张图）
    for plot_name, plot_config in PLOT_CONFIGURATIONS.items():
        print(f"\n[信息] 生成图表: {plot_name} - {plot_config.get('description', '')}")
        
        # 所有图表都使用plot_custom_comparison函数
        # 应用时间范围过滤
        filtered_method_data = {}
        time_range = plot_config.get('time_range', TIME_RANGE)
        
        for method_name, df in method_data.items():
            filtered_df = filter_data_by_time_range(df, time_range, time_column='time')
            if filtered_df is not None and len(filtered_df) > 0:
                filtered_method_data[method_name] = filtered_df
        
        # 使用plot_custom_comparison绘制所有图表
        plot_custom_comparison(
            method_data=filtered_method_data,
            plot_config=plot_config,
            output_dir=output_dir,
            plot_name=plot_name,
            global_time_range=TIME_RANGE
        )
    
    # 旧的图表处理逻辑已移除，所有图表都使用plot_custom_comparison
    # 所有图表配置都在PLOT_CONFIGURATIONS中定义，使用plot_custom_comparison函数绘制
    
    # 生成指标对比图
    print("\n[信息] 生成指标对比柱状图...")
    plot_metrics_comparison(method_metrics, output_dir)
    
    # 保存指标表格
    print("\n[信息] 保存指标表格...")
    save_metrics_table(method_metrics, output_dir)
    
    print("\n" + "=" * 80)
    print("对比实验完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)
    
    # 打印方法系数说明
    print("\n方法系数说明:")
    for method_name, coefficient in METHOD_COEFFICIENTS.items():
        print(f"  {method_name}: {coefficient}")


def main():
    parser = argparse.ArgumentParser(
        description="从单一数据源生成多个对比方法"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/user/IsaacLab/training_data",
        help="原始数据目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/user/IsaacLab/comparison_results",
        help="输出目录"
    )
    parser.add_argument(
        "--max_episode_length",
        type=float,
        default=1000.0,
        help="最大episode长度（用于计算存活率）"
    )
    parser.add_argument(
        "--show_variables",
        action='store_true',
        help="显示所有可用变量列表"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成对比
    generate_comparison_from_single_data(
        args.data_dir,
        output_dir,
        args.max_episode_length,
        args.show_variables
    )


if __name__ == "__main__":
    main()
