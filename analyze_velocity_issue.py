#!/usr/bin/env python3
"""
分析速度对应不好的问题
"""

import numpy as np
import os
import glob

def analyze_velocity_issue():
    """分析速度对应不好的问题"""
    
    print("=== 速度对应问题分析 ===")
    
    # 检查数据文件
    data_dir = "/home/user/IsaacLab/comparison_data"
    files = glob.glob(f"{data_dir}/env_0_step_*.npz")
    
    if not files:
        print("没有找到对比数据文件！")
        return
    
    # 加载第一个文件进行分析
    first_file = sorted(files)[0]
    print(f"分析文件: {first_file}")
    
    data = np.load(first_file)
    
    # 提取数据
    isaaclab_eta = data['isaaclab_eta_history']
    isaaclab_nu = data['isaaclab_nu_history']
    calculated_eta = data['calculated_eta_history']
    calculated_nu = data['calculated_nu_history']
    
    print(f"\n数据形状:")
    print(f"  isaaclab_eta: {isaaclab_eta.shape}")
    print(f"  isaaclab_nu: {isaaclab_nu.shape}")
    print(f"  calculated_eta: {calculated_eta.shape}")
    print(f"  calculated_nu: {calculated_nu.shape}")
    
    # 分析位置对应情况
    print(f"\n=== 位置对应分析 ===")
    eta_diff = isaaclab_eta - calculated_eta
    eta_norm = np.linalg.norm(eta_diff, axis=1)
    
    print(f"位置差别统计:")
    print(f"  平均差别范数: {np.mean(eta_norm):.6f}")
    print(f"  最大差别范数: {np.max(eta_norm):.6f}")
    print(f"  标准差: {np.std(eta_norm):.6f}")
    
    # 分析速度对应情况
    print(f"\n=== 速度对应分析 ===")
    nu_diff = isaaclab_nu - calculated_nu
    nu_norm = np.linalg.norm(nu_diff, axis=1)
    
    print(f"速度差别统计:")
    print(f"  平均差别范数: {np.mean(nu_norm):.6f}")
    print(f"  最大差别范数: {np.max(nu_norm):.6f}")
    print(f"  标准差: {np.std(nu_norm):.6f}")
    
    # 详细分析速度的各个分量
    print(f"\n=== 速度分量分析 ===")
    component_names = ['u (surge)', 'v (sway)', 'w (heave)', 'p (roll_rate)', 'q (pitch_rate)', 'r (yaw_rate)']
    
    for i, name in enumerate(component_names):
        diff_i = nu_diff[:, i]
        print(f"{name}:")
        print(f"  平均差别: {np.mean(diff_i):.6f}")
        print(f"  最大差别: {np.max(np.abs(diff_i)):.6f}")
        print(f"  标准差: {np.std(diff_i):.6f}")
    
    # 检查是否有系统性偏差
    print(f"\n=== 系统性偏差分析 ===")
    for i, name in enumerate(component_names):
        diff_i = nu_diff[:, i]
        mean_diff = np.mean(diff_i)
        if abs(mean_diff) > 0.001:  # 如果平均差别大于0.001
            print(f"{name} 存在系统性偏差: {mean_diff:.6f}")
        else:
            print(f"{name} 无明显系统性偏差")
    
    # 检查速度的数值范围
    print(f"\n=== 速度数值范围分析 ===")
    print(f"IsaacLab速度范围:")
    for i, name in enumerate(component_names):
        min_val = np.min(isaaclab_nu[:, i])
        max_val = np.max(isaaclab_nu[:, i])
        print(f"  {name}: [{min_val:.6f}, {max_val:.6f}]")
    
    print(f"积分计算速度范围:")
    for i, name in enumerate(component_names):
        min_val = np.min(calculated_nu[:, i])
        max_val = np.max(calculated_nu[:, i])
        print(f"  {name}: [{min_val:.6f}, {max_val:.6f}]")
    
    # 检查速度的变化趋势
    print(f"\n=== 速度变化趋势分析 ===")
    for i, name in enumerate(component_names):
        isaaclab_trend = np.diff(isaaclab_nu[:, i])
        calculated_trend = np.diff(calculated_nu[:, i])
        trend_diff = isaaclab_trend - calculated_trend
        
        print(f"{name} 变化趋势差别:")
        print(f"  平均趋势差别: {np.mean(trend_diff):.6f}")
        print(f"  最大趋势差别: {np.max(np.abs(trend_diff)):.6f}")
    
    return True

if __name__ == "__main__":
    analyze_velocity_issue()

