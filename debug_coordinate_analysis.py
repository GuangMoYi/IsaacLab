#!/usr/bin/env python3
"""
调试坐标系分析 - 分析IsaacLab与船舶动力学的坐标系差异
"""

import numpy as np
import os
import glob

def analyze_coordinate_system():
    """分析坐标系问题"""
    
    print("=== 坐标系差异分析 ===")
    
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
    
    print(f"\n=== 坐标系定义分析 ===")
    print(f"IsaacLab坐标系:")
    print(f"  root_lin_vel_b: [u, v, w] - 刚体坐标系线速度")
    print(f"  root_ang_vel_b: [p, q, r] - 刚体坐标系角速度")
    print(f"  组合: [u, v, w, p, q, r]")
    
    print(f"\n船舶动力学坐标系:")
    print(f"  nu = [u, v, w, p, q, r]")
    print(f"  其中: u=surge(前), v=sway(右), w=heave(下)")
    print(f"        p=roll_rate, q=pitch_rate, r=yaw_rate")
    
    # 分析速度的数值特征
    print(f"\n=== 速度数值特征分析 ===")
    component_names = ['u (surge)', 'v (sway)', 'w (heave)', 'p (roll_rate)', 'q (pitch_rate)', 'r (yaw_rate)']
    
    for i, name in enumerate(component_names):
        isaaclab_values = isaaclab_nu[:, i]
        calculated_values = calculated_nu[:, i]
        
        print(f"\n{name}:")
        print(f"  IsaacLab范围: [{np.min(isaaclab_values):.6f}, {np.max(isaaclab_values):.6f}]")
        print(f"  积分计算范围: [{np.min(calculated_values):.6f}, {np.max(calculated_values):.6f}]")
        print(f"  平均差异: {np.mean(isaaclab_values - calculated_values):.6f}")
        print(f"  最大差异: {np.max(np.abs(isaaclab_values - calculated_values)):.6f}")
        
        # 检查是否有系统性偏差
        mean_diff = np.mean(isaaclab_values - calculated_values)
        if abs(mean_diff) > 0.001:
            print(f"  ⚠️  存在系统性偏差: {mean_diff:.6f}")
        else:
            print(f"  ✅ 无明显系统性偏差")
    
    # 分析速度的变化趋势
    print(f"\n=== 速度变化趋势分析 ===")
    for i, name in enumerate(component_names):
        isaaclab_diff = np.diff(isaaclab_nu[:, i])
        calculated_diff = np.diff(calculated_nu[:, i])
        trend_diff = isaaclab_diff - calculated_diff
        
        print(f"{name} 变化趋势:")
        print(f"  平均趋势差异: {np.mean(trend_diff):.6f}")
        print(f"  最大趋势差异: {np.max(np.abs(trend_diff)):.6f}")
        print(f"  趋势差异标准差: {np.std(trend_diff):.6f}")
    
    # 分析位置与速度的对应关系
    print(f"\n=== 位置与速度对应关系分析 ===")
    print(f"位置差异统计:")
    eta_diff = isaaclab_eta - calculated_eta
    eta_norm = np.linalg.norm(eta_diff, axis=1)
    print(f"  平均位置差异范数: {np.mean(eta_norm):.6f}")
    print(f"  最大位置差异范数: {np.max(eta_norm):.6f}")
    
    print(f"速度差异统计:")
    nu_diff = isaaclab_nu - calculated_nu
    nu_norm = np.linalg.norm(nu_diff, axis=1)
    print(f"  平均速度差异范数: {np.mean(nu_norm):.6f}")
    print(f"  最大速度差异范数: {np.max(nu_norm):.6f}")
    
    # 分析差异的演化
    print(f"\n=== 差异演化分析 ===")
    print(f"位置差异演化:")
    print(f"  初始位置差异: {np.linalg.norm(eta_diff[0]):.6f}")
    print(f"  最终位置差异: {np.linalg.norm(eta_diff[-1]):.6f}")
    print(f"  位置差异增长率: {(np.linalg.norm(eta_diff[-1]) - np.linalg.norm(eta_diff[0])):.6f}")
    
    print(f"速度差异演化:")
    print(f"  初始速度差异: {np.linalg.norm(nu_diff[0]):.6f}")
    print(f"  最终速度差异: {np.linalg.norm(nu_diff[-1]):.6f}")
    print(f"  速度差异增长率: {(np.linalg.norm(nu_diff[-1]) - np.linalg.norm(nu_diff[0])):.6f}")
    
    # 检查是否有坐标系转换问题
    print(f"\n=== 坐标系转换问题检查 ===")
    print(f"如果存在坐标系转换问题，应该看到:")
    print(f"  1. 线速度分量存在系统性偏差")
    print(f"  2. 角速度分量基本正常")
    print(f"  3. 差异随时间累积")
    
    # 检查线速度与角速度的差异模式
    lin_vel_diff = nu_diff[:, :3]  # 线速度差异
    ang_vel_diff = nu_diff[:, 3:]  # 角速度差异
    
    print(f"\n线速度差异统计:")
    print(f"  平均线速度差异范数: {np.mean(np.linalg.norm(lin_vel_diff, axis=1)):.6f}")
    print(f"  最大线速度差异范数: {np.max(np.linalg.norm(lin_vel_diff, axis=1)):.6f}")
    
    print(f"角速度差异统计:")
    print(f"  平均角速度差异范数: {np.mean(np.linalg.norm(ang_vel_diff, axis=1)):.6f}")
    print(f"  最大角速度差异范数: {np.max(np.linalg.norm(ang_vel_diff, axis=1)):.6f}")
    
    return True

if __name__ == "__main__":
    analyze_coordinate_system()

