#!/usr/bin/env python3
"""
调试坐标系问题的简化脚本
"""

import numpy as np

def debug_coordinate_system():
    """调试坐标系问题"""
    
    print("=== 坐标系调试分析 ===")
    
    # 模拟IsaacLab的速度数据
    print("\n1. IsaacLab速度数据格式:")
    print("   root_lin_vel_b: [u, v, w] - 刚体坐标系线速度")
    print("   root_ang_vel_b: [p, q, r] - 刚体坐标系角速度")
    print("   组合: [u, v, w, p, q, r]")
    
    # 模拟船舶控制系统的速度定义
    print("\n2. 船舶控制系统速度定义:")
    print("   nu = [u, v, w, p, q, r]")
    print("   其中: u=surge(前), v=sway(右), w=heave(下)")
    print("        p=roll_rate, q=pitch_rate, r=yaw_rate")
    
    # 检查坐标系一致性
    print("\n3. 坐标系一致性检查:")
    print("   IsaacLab的刚体坐标系是否与船舶动力学的船体坐标系一致？")
    print("   - 如果一致：速度应该直接对应")
    print("   - 如果不一致：需要坐标系转换")
    
    # 模拟数据
    print("\n4. 模拟数据对比:")
    
    # 模拟IsaacLab输出
    isaaclab_nu = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.3])  # [u, v, w, p, q, r]
    print(f"   IsaacLab速度: {isaaclab_nu}")
    
    # 模拟船舶控制系统期望的输入
    vessel_nu = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.3])  # 假设一致
    print(f"   船舶控制系统输入: {vessel_nu}")
    
    # 检查差异
    diff = isaaclab_nu - vessel_nu
    print(f"   差异: {diff}")
    print(f"   差异范数: {np.linalg.norm(diff):.6f}")
    
    # 分析可能的问题
    print("\n5. 可能的问题分析:")
    print("   a) 坐标系定义不一致")
    print("   b) 速度分量顺序不同")
    print("   c) 单位不一致")
    print("   d) 参考系不同")
    
    # 建议的调试步骤
    print("\n6. 建议的调试步骤:")
    print("   1. 检查IsaacLab的刚体坐标系定义")
    print("   2. 检查船舶动力学的船体坐标系定义")
    print("   3. 比较两个坐标系的差异")
    print("   4. 如果需要，添加坐标系转换")
    
    return True

if __name__ == "__main__":
    debug_coordinate_system()
