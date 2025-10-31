#!/usr/bin/env python3
"""
性能分析脚本 - 找出真正的性能瓶颈
"""

import numpy as np
import time
import sys
import os

# 添加路径以导入vessels模块
sys.path.append('/home/user/IsaacLab/source/isaaclab/isaaclab/envs/mdp')
from vessels import VesselControlSystem

def test_vessel_system_performance():
    """测试船舶控制系统的性能"""
    
    print("=== 船舶控制系统性能分析 ===")
    
    # 创建船舶控制系统
    target_position = [10, 10, 0.8 * np.pi]
    initial_eta = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    initial_nu = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    
    vessel_system = VesselControlSystem(
        target_position=target_position,
        initial_eta=initial_eta,
        initial_nu=initial_nu,
        dt=0.01
    )
    
    print(f"船舶控制系统创建完成")
    
    # 测试不同环境数量的性能
    test_cases = [1, 10, 50, 100, 500, 1000]
    
    for num_envs in test_cases:
        print(f"\n--- 测试 {num_envs} 个环境 ---")
        
        # 生成随机状态数据
        poses = np.random.randn(num_envs, 6) * 0.1
        nus = np.random.randn(num_envs, 6) * 0.1
        current_time = 0.0
        
        # 测试方法1：每个环境独立创建系统
        start_time = time.time()
        for i in range(num_envs):
            # 模拟每个环境都有独立的系统
            vessel_system.eta[:] = poses[i]
            vessel_system.nu[:] = nus[i]
            acc, eta_dot = vessel_system.step(poses[i], nus[i], current_time)
        method1_time = time.time() - start_time
        
        # 测试方法2：共享系统但循环调用
        start_time = time.time()
        for i in range(num_envs):
            vessel_system.eta[:] = poses[i]
            vessel_system.nu[:] = nus[i]
            acc, eta_dot = vessel_system.step(poses[i], nus[i], current_time)
        method2_time = time.time() - start_time
        
        print(f"方法1 (独立系统): {method1_time*1000:.2f}ms")
        print(f"方法2 (共享系统): {method2_time*1000:.2f}ms")
        print(f"单次step平均耗时: {method2_time*1000/num_envs:.3f}ms")
        
        if num_envs >= 100:
            print(f"预计1024环境耗时: {method2_time*1000/num_envs*1024:.0f}ms = {method2_time*1000/num_envs*1024/1000:.1f}秒")

def test_step_breakdown():
    """分析step方法内部各部分的耗时"""
    
    print("\n=== Step方法内部耗时分析 ===")
    
    # 创建船舶控制系统
    target_position = [10, 10, 0.8 * np.pi]
    initial_eta = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    initial_nu = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    
    vessel_system = VesselControlSystem(
        target_position=target_position,
        initial_eta=initial_eta,
        initial_nu=initial_nu,
        dt=0.01
    )
    
    # 测试数据
    current_eta = np.array([1, 2, 0.1, 0.05, 0.02, 0.1])
    current_nu = np.array([0.1, 0.2, 0.01, 0.05, 0.02, 0.1])
    current_time = 1.0
    
    # 分析各部分耗时
    times = {}
    
    # 1. 数据类型转换
    start = time.time()
    if hasattr(current_eta, 'cpu'):
        current_eta_np = current_eta.detach().cpu().numpy()
        current_nu_np = current_nu.detach().cpu().numpy()
    else:
        current_eta_np = current_eta
        current_nu_np = current_nu
    times['data_conversion'] = time.time() - start
    
    # 2. 状态更新
    start = time.time()
    vessel_system.eta[:] = current_eta_np
    vessel_system.nu[:] = current_nu_np
    times['state_update'] = time.time() - start
    
    # 3. 参考轨迹计算
    start = time.time()
    eta_r, eta_r_dot = vessel_system.reference[0:3], vessel_system.reference[3:6]
    vessel_system.eta_r_ddot = vessel_system.reference_model_dynamics(eta_r, eta_r_dot)
    reference_dot = np.concatenate([eta_r_dot, vessel_system.eta_r_ddot])
    times['reference_trajectory'] = time.time() - start
    
    # 4. 观测器计算
    start = time.time()
    y_hat = vessel_system.x_hat[0:3]
    vessel_system.x_hat = np.concatenate([vessel_system.eta[[0, 1, 5]], vessel_system.nu[[0, 1, 5]]])
    vessel_system.nu_hat = vessel_system.nu[[0, 1, 5]]
    vessel_system.b_hat.fill(0)
    vessel_system.xi_hat.fill(0)
    times['observer'] = time.time() - start
    
    # 5. 控制器计算
    start = time.time()
    current_control_acceleration = vessel_system.controller_acceleration(eta_r, vessel_system.x_hat, vessel_system.b_hat, 
                                                                       current_eta_np, current_nu_np)
    times['controller'] = time.time() - start
    
    # 6. 波浪载荷计算
    start = time.time()
    wave_loads = vessel_system.generate_wave_loads_jonswap(current_time)
    current_control_acceleration += vessel_system.inv_M @ wave_loads
    times['wave_loads'] = time.time() - start
    
    # 7. 位置导数计算
    start = time.time()
    R = vessel_system.Rzyx(vessel_system.eta[3:6])
    T_mat = vessel_system.T_Theta(vessel_system.eta[3:6])
    vessel_system._temp_eta_dot[:3] = R @ vessel_system.nu[:3]
    vessel_system._temp_eta_dot[3:6] = T_mat @ vessel_system.nu[3:6]
    eta_dot = vessel_system._temp_eta_dot.copy()
    times['position_derivative'] = time.time() - start
    
    # 8. 状态更新
    start = time.time()
    vessel_system.eta += eta_dot * vessel_system.dt
    vessel_system.nu += current_control_acceleration * vessel_system.dt
    vessel_system.reference += reference_dot * vessel_system.dt
    times['final_state_update'] = time.time() - start
    
    # 输出结果
    total_time = sum(times.values())
    print(f"总耗时: {total_time*1000:.3f}ms")
    print()
    
    for component, duration in times.items():
        percentage = duration / total_time * 100
        print(f"{component:20s}: {duration*1000:6.3f}ms ({percentage:5.1f}%)")

if __name__ == "__main__":
    test_vessel_system_performance()
    test_step_breakdown()
