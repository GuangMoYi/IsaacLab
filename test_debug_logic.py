#!/usr/bin/env python3
"""
测试调试逻辑的简化脚本
"""

import numpy as np

def test_debug_logic():
    """测试调试逻辑"""
    
    print("=== 调试逻辑测试 ===")
    
    # 模拟环境ID
    env_id = 0
    
    # 模拟comp_data
    comp_data = {
        'step_count': 0,
        'isaaclab_eta_history': [],
        'isaaclab_nu_history': [],
        'calculated_eta_history': [],
        'calculated_nu_history': []
    }
    
    # 模拟IsaacLab数据
    isaaclab_eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    isaaclab_nu = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # 模拟积分计算
    calculated_eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    calculated_nu = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # 测试零加速度
    zero_acc = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print(f"初始状态:")
    print(f"  IsaacLab位置: {isaaclab_eta[:3]}")
    print(f"  IsaacLab速度: {isaaclab_nu[:3]}")
    print(f"  积分计算位置: {calculated_eta[:3]}")
    print(f"  积分计算速度: {calculated_nu[:3]}")
    print(f"  零加速度: {zero_acc}")
    
    # 模拟100步
    for step in range(1, 101):
        comp_data['step_count'] = step
        
        # 零加速度下，理论上速度应该保持不变
        # 但IsaacLab可能有重力、阻尼等外力影响
        
        # 模拟IsaacLab的物理仿真（可能有重力影响）
        if step > 1:
            # 模拟重力影响（z方向）
            isaaclab_nu[2] -= 0.001  # 重力加速度影响
            isaaclab_eta[2] += isaaclab_nu[2] * 0.02  # 位置更新
        
        # 积分计算（零加速度）
        calculated_nu += zero_acc * 0.02  # 速度更新
        calculated_eta += calculated_nu * 0.02  # 位置更新
        
        # 每10步打印一次
        if step % 10 == 0:
            diff_eta = isaaclab_eta - calculated_eta
            diff_nu = isaaclab_nu - calculated_nu
            pos_norm = np.linalg.norm(diff_eta[:3])
            vel_norm = np.linalg.norm(diff_nu[:3])
            
            print(f"\n第 {step} 步:")
            print(f"  IsaacLab位置: {isaaclab_eta[:3]}")
            print(f"  积分计算位置: {calculated_eta[:3]}")
            print(f"  位置差异: {diff_eta[:3]}")
            print(f"  位置差异范数: {pos_norm:.6f}")
            
            print(f"  IsaacLab速度: {isaaclab_nu[:3]}")
            print(f"  积分计算速度: {calculated_nu[:3]}")
            print(f"  速度差异: {diff_nu[:3]}")
            print(f"  速度差异范数: {vel_norm:.6f}")
    
    print(f"\n=== 测试完成 ===")
    print(f"最终IsaacLab位置: {isaaclab_eta[:3]}")
    print(f"最终积分计算位置: {calculated_eta[:3]}")
    print(f"最终IsaacLab速度: {isaaclab_nu[:3]}")
    print(f"最终积分计算速度: {calculated_nu[:3]}")

if __name__ == "__main__":
    test_debug_logic()

