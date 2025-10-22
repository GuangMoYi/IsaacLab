#!/usr/bin/env python3
"""
测试常数加速度的调试脚本
"""

import numpy as np
import matplotlib.pyplot as plt

def test_constant_acceleration():
    """测试常数加速度的积分计算"""
    
    print("=== 常数加速度测试 ===")
    
    # 参数设置
    dt = 0.02  # 时间步长
    total_time = 10.0  # 总时间
    steps = int(total_time / dt)
    
    # 常数加速度
    constant_acc = np.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.0])  # [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]
    
    # 初始状态
    eta = np.zeros(6)  # 位置
    nu = np.zeros(6)   # 速度
    
    # 存储历史数据
    eta_history = []
    nu_history = []
    time_history = []
    
    print(f"初始状态:")
    print(f"  位置: {eta}")
    print(f"  速度: {nu}")
    print(f"  加速度: {constant_acc}")
    print(f"  时间步长: {dt}")
    print(f"  总步数: {steps}")
    
    # 积分计算
    for i in range(steps):
        # 存储当前状态
        eta_history.append(eta.copy())
        nu_history.append(nu.copy())
        time_history.append(i * dt)
        
        # 积分更新
        eta += nu * dt  # 位置更新
        nu += constant_acc * dt  # 速度更新
        
        # 每100步打印一次
        if i % 100 == 0:
            print(f"\n第 {i} 步 (时间: {i*dt:.2f}s):")
            print(f"  位置: {eta[:3]}")
            print(f"  速度: {nu[:3]}")
            print(f"  理论位置: {constant_acc[:3] * (i*dt)**2 / 2}")
            print(f"  理论速度: {constant_acc[:3] * (i*dt)}")
    
    # 最终结果
    print(f"\n=== 最终结果 ===")
    print(f"最终位置: {eta}")
    print(f"最终速度: {nu}")
    print(f"理论最终位置: {constant_acc * (total_time)**2 / 2}")
    print(f"理论最终速度: {constant_acc * total_time}")
    
    # 计算误差
    theoretical_eta = constant_acc * (total_time)**2 / 2
    theoretical_nu = constant_acc * total_time
    
    eta_error = np.linalg.norm(eta - theoretical_eta)
    nu_error = np.linalg.norm(nu - theoretical_nu)
    
    print(f"\n=== 误差分析 ===")
    print(f"位置误差: {eta_error:.6f}")
    print(f"速度误差: {nu_error:.6f}")
    
    return eta_history, nu_history, time_history

if __name__ == "__main__":
    eta_history, nu_history, time_history = test_constant_acceleration()

