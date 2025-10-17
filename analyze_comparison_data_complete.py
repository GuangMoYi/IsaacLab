#!/usr/bin/env python3
"""
完整分析IsaacLab对比数据 - 按时间顺序合并所有文件
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def analyze_complete_comparison_data():
    """完整分析对比数据 - 按时间顺序合并所有文件"""
    
    # 数据目录
    data_dir = "/home/user/IsaacLab/comparison_data/202510171100(Hs=1)"
    
    # 获取所有数据文件（包括子目录）
    # files = glob.glob(f"{data_dir}/**/env_*_step_*.npz", recursive=True)
    files = glob.glob(f"{data_dir}/env_0_step_*.npz", recursive=True)
    files.sort()
    
    if not files:
        print("没有找到对比数据文件！")
        return
    
    print(f"找到 {len(files)} 个数据文件")
    
    # 只分析第一个环境（env_id=0）的数据
    env_data = {}
    
    for file in files:
        # 解析文件名
        filename = os.path.basename(file)
        parts = filename.replace('.npz', '').split('_')
        env_id = int(parts[1])
        step = int(parts[3])
        
        # 只处理第一个环境的数据
        if env_id != 0:
            continue
        
        # 加载数据
        data = np.load(file)
        
        if env_id not in env_data:
            env_data[env_id] = {
                'isaaclab_eta': [],
                'isaaclab_nu': [],
                'calculated_eta': [],
                'calculated_nu': [],
                'differences_eta': [],
                'differences_nu': [],
                'time_steps': [],
                'max_step': 0
            }
        
        # 提取数据
        isaaclab_eta = data['isaaclab_eta_history']  # [step_count, 6]
        isaaclab_nu = data['isaaclab_nu_history']    # [step_count, 6]
        calculated_eta = data['calculated_eta_history']  # [step_count, 6]
        calculated_nu = data['calculated_nu_history']    # [step_count, 6]
        
        # 计算差别
        diff_eta = isaaclab_eta - calculated_eta
        diff_nu = isaaclab_nu - calculated_nu
        
        # 生成时间步
        time_steps = np.arange(1, len(isaaclab_eta) + 1)
        
        # 只保留最新的数据（因为每个文件都包含完整历史）
        if step > env_data[env_id]['max_step']:
            env_data[env_id]['isaaclab_eta'] = isaaclab_eta.tolist()
            env_data[env_id]['isaaclab_nu'] = isaaclab_nu.tolist()
            env_data[env_id]['calculated_eta'] = calculated_eta.tolist()
            env_data[env_id]['calculated_nu'] = calculated_nu.tolist()
            env_data[env_id]['differences_eta'] = diff_eta.tolist()
            env_data[env_id]['differences_nu'] = diff_nu.tolist()
            env_data[env_id]['time_steps'] = time_steps.tolist()
            env_data[env_id]['max_step'] = step
    
    # 转换为numpy数组
    for env_id in env_data:
        for key in ['isaaclab_eta', 'isaaclab_nu', 'calculated_eta', 'calculated_nu', 'differences_eta', 'differences_nu', 'time_steps']:
            env_data[env_id][key] = np.array(env_data[env_id][key])
    
    # 分析结果（只显示第一个环境）
    print("\n=== 完整对比分析结果（环境0） ===")
    
    if 0 in env_data:
        data = env_data[0]
        print(f"\n环境 0:")
        print(f"  总步数: {len(data['time_steps'])}")
        print(f"  时间范围: {data['time_steps'][0]} - {data['time_steps'][-1]}")
        
        # 位置差别统计
        diff_eta = data['differences_eta']
        print(f"  位置差别统计:")
        print(f"    X方向: 均值={np.mean(diff_eta[:, 0]):.6f}, 标准差={np.std(diff_eta[:, 0]):.6f}, 最大值={np.max(np.abs(diff_eta[:, 0])):.6f}")
        print(f"    Y方向: 均值={np.mean(diff_eta[:, 1]):.6f}, 标准差={np.std(diff_eta[:, 1]):.6f}, 最大值={np.max(np.abs(diff_eta[:, 1])):.6f}")
        print(f"    Z方向: 均值={np.mean(diff_eta[:, 2]):.6f}, 标准差={np.std(diff_eta[:, 2]):.6f}, 最大值={np.max(np.abs(diff_eta[:, 2])):.6f}")
        print(f"    Roll: 均值={np.mean(diff_eta[:, 3]):.6f}, 标准差={np.std(diff_eta[:, 3]):.6f}, 最大值={np.max(np.abs(diff_eta[:, 3])):.6f}")
        print(f"    Pitch: 均值={np.mean(diff_eta[:, 4]):.6f}, 标准差={np.std(diff_eta[:, 4]):.6f}, 最大值={np.max(np.abs(diff_eta[:, 4])):.6f}")
        print(f"    Yaw: 均值={np.mean(diff_eta[:, 5]):.6f}, 标准差={np.std(diff_eta[:, 5]):.6f}, 最大值={np.max(np.abs(diff_eta[:, 5])):.6f}")
        
        # 速度差别统计
        diff_nu = data['differences_nu']
        print(f"  速度差别统计:")
        print(f"    X方向: 均值={np.mean(diff_nu[:, 0]):.6f}, 标准差={np.std(diff_nu[:, 0]):.6f}, 最大值={np.max(np.abs(diff_nu[:, 0])):.6f}")
        print(f"    Y方向: 均值={np.mean(diff_nu[:, 1]):.6f}, 标准差={np.std(diff_nu[:, 1]):.6f}, 最大值={np.max(np.abs(diff_nu[:, 1])):.6f}")
        print(f"    Z方向: 均值={np.mean(diff_nu[:, 2]):.6f}, 标准差={np.std(diff_nu[:, 2]):.6f}, 最大值={np.max(np.abs(diff_nu[:, 2])):.6f}")
        print(f"    Roll: 均值={np.mean(diff_nu[:, 3]):.6f}, 标准差={np.std(diff_nu[:, 3]):.6f}, 最大值={np.max(np.abs(diff_nu[:, 3])):.6f}")
        print(f"    Pitch: 均值={np.mean(diff_nu[:, 4]):.6f}, 标准差={np.std(diff_nu[:, 4]):.6f}, 最大值={np.max(np.abs(diff_nu[:, 4])):.6f}")
        print(f"    Yaw: 均值={np.mean(diff_nu[:, 5]):.6f}, 标准差={np.std(diff_nu[:, 5]):.6f}, 最大值={np.max(np.abs(diff_nu[:, 5])):.6f}")
        
        # 差别范数
        eta_norm = np.linalg.norm(diff_eta, axis=1)
        nu_norm = np.linalg.norm(diff_nu, axis=1)
        print(f"  位置差别范数: 均值={np.mean(eta_norm):.6f}, 最大值={np.max(eta_norm):.6f}")
        print(f"  速度差别范数: 均值={np.mean(nu_norm):.6f}, 最大值={np.max(nu_norm):.6f}")
    else:
        print("没有找到环境0的数据！")
    
    # 绘制图形
    plot_complete_comparison_results(env_data)
    
    return env_data

def plot_complete_comparison_results(env_data):
    """绘制完整的对比结果图形（只绘制环境0）"""
    
    # 只绘制第一个环境（env_id=0）
    if 0 not in env_data:
        print("没有找到环境0的数据，无法绘图！")
        return
    
    data = env_data[0]
    
    print(f"\n绘制环境 0 的完整对比图")
    print(f"数据点数: {len(data['time_steps'])}")
    print(f"时间范围: {data['time_steps'][0]} - {data['time_steps'][-1]}")
    
    # 创建位置对比图
    plt.figure(figsize=(15, 10))
    
    # 位置对比 - 前3个自由度
    plt.subplot(2, 3, 1)
    plt.plot(data['time_steps'], data['isaaclab_eta'][:, 0], 'r-', label='IsaacLab X Position', linewidth=1.5, alpha=0.8)
    plt.plot(data['time_steps'], data['calculated_eta'][:, 0], 'b--', label='Calculated X Position', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('X Position (m)')
    plt.title('X Position Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(data['time_steps'], data['isaaclab_eta'][:, 1], 'r-', label='IsaacLab Y Position', linewidth=1.5, alpha=0.8)
    plt.plot(data['time_steps'], data['calculated_eta'][:, 1], 'b--', label='Calculated Y Position', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(data['time_steps'], data['isaaclab_eta'][:, 2], 'r-', label='IsaacLab Z Position', linewidth=1.5, alpha=0.8)
    plt.plot(data['time_steps'], data['calculated_eta'][:, 2], 'b--', label='Calculated Z Position', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Z Position (m)')
    plt.title('Z Position Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 位置对比 - 后3个自由度（角度）
    plt.subplot(2, 3, 4)
    plt.plot(data['time_steps'], data['isaaclab_eta'][:, 3], 'r-', label='IsaacLab Roll Angle', linewidth=1.5, alpha=0.8)
    plt.plot(data['time_steps'], data['calculated_eta'][:, 3], 'b--', label='Calculated Roll Angle', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Roll Angle (rad)')
    plt.title('Roll Angle Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.plot(data['time_steps'], data['isaaclab_eta'][:, 4], 'r-', label='IsaacLab Pitch Angle', linewidth=1.5, alpha=0.8)
    plt.plot(data['time_steps'], data['calculated_eta'][:, 4], 'b--', label='Calculated Pitch Angle', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Pitch Angle (rad)')
    plt.title('Pitch Angle Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    plt.plot(data['time_steps'], data['isaaclab_eta'][:, 5], 'r-', label='IsaacLab Yaw Angle', linewidth=1.5, alpha=0.8)
    plt.plot(data['time_steps'], data['calculated_eta'][:, 5], 'b--', label='Calculated Yaw Angle', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Yaw Angle (rad)')
    plt.title('Yaw Angle Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/user/IsaacLab/complete_position_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建速度对比图
    plt.figure(figsize=(15, 10))
    
    # 速度对比 - 前3个自由度
    plt.subplot(2, 3, 1)
    plt.plot(data['time_steps'], data['isaaclab_nu'][:, 0], 'r-', label='IsaacLab X Velocity', linewidth=1.5, alpha=0.8)
    plt.plot(data['time_steps'], data['calculated_nu'][:, 0], 'b--', label='Calculated X Velocity', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('X Velocity (m/s)')
    plt.title('X Velocity Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(data['time_steps'], data['isaaclab_nu'][:, 1], 'r-', label='IsaacLab Y Velocity', linewidth=1.5, alpha=0.8)
    plt.plot(data['time_steps'], data['calculated_nu'][:, 1], 'b--', label='Calculated Y Velocity', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Y Velocity (m/s)')
    plt.title('Y Velocity Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(data['time_steps'], data['isaaclab_nu'][:, 2], 'r-', label='IsaacLab Z Velocity', linewidth=1.5, alpha=0.8)
    plt.plot(data['time_steps'], data['calculated_nu'][:, 2], 'b--', label='Calculated Z Velocity', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Z Velocity (m/s)')
    plt.title('Z Velocity Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 速度对比 - 后3个自由度（角速度）
    plt.subplot(2, 3, 4)
    plt.plot(data['time_steps'], data['isaaclab_nu'][:, 3], 'r-', label='IsaacLab Roll Rate', linewidth=1.5, alpha=0.8)
    plt.plot(data['time_steps'], data['calculated_nu'][:, 3], 'b--', label='Calculated Roll Rate', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Roll Rate (rad/s)')
    plt.title('Roll Rate Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.plot(data['time_steps'], data['isaaclab_nu'][:, 4], 'r-', label='IsaacLab Pitch Rate', linewidth=1.5, alpha=0.8)
    plt.plot(data['time_steps'], data['calculated_nu'][:, 4], 'b--', label='Calculated Pitch Rate', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Pitch Rate (rad/s)')
    plt.title('Pitch Rate Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    plt.plot(data['time_steps'], data['isaaclab_nu'][:, 5], 'r-', label='IsaacLab Yaw Rate', linewidth=1.5, alpha=0.8)
    plt.plot(data['time_steps'], data['calculated_nu'][:, 5], 'b--', label='Calculated Yaw Rate', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Yaw Rate (rad/s)')
    plt.title('Yaw Rate Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/user/IsaacLab/complete_velocity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建差别图
    plt.figure(figsize=(15, 10))
    
    # 位置差别
    plt.subplot(2, 3, 1)
    diff_eta = data['differences_eta']
    plt.plot(data['time_steps'], diff_eta[:, 0], 'g-', label='X Position Difference', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('X Position Difference (m)')
    plt.title('X Position Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(data['time_steps'], diff_eta[:, 1], 'g-', label='Y Position Difference', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Y Position Difference (m)')
    plt.title('Y Position Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(data['time_steps'], diff_eta[:, 5], 'g-', label='Yaw Angle Difference', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Yaw Angle Difference (rad)')
    plt.title('Yaw Angle Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 速度差别
    plt.subplot(2, 3, 4)
    diff_nu = data['differences_nu']
    plt.plot(data['time_steps'], diff_nu[:, 0], 'orange', label='X Velocity Difference', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('X Velocity Difference (m/s)')
    plt.title('X Velocity Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.plot(data['time_steps'], diff_nu[:, 1], 'orange', label='Y Velocity Difference', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Y Velocity Difference (m/s)')
    plt.title('Y Velocity Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    plt.plot(data['time_steps'], diff_nu[:, 5], 'orange', label='Yaw Rate Difference', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Yaw Rate Difference (rad/s)')
    plt.title('Yaw Rate Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/user/IsaacLab/complete_differences_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n完整图形已保存到:")
    print(f"  位置对比: /home/user/IsaacLab/complete_position_comparison.png")
    print(f"  速度对比: /home/user/IsaacLab/complete_velocity_comparison.png")
    print(f"  差别对比: /home/user/IsaacLab/complete_differences_comparison.png")

if __name__ == "__main__":
    env_data = analyze_complete_comparison_data()
    print("\n完整分析完成！")
