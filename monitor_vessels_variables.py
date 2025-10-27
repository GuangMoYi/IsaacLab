#!/usr/bin/env python3
"""
监控vessels中所有变量的变化
记录训练过程中所有关键变量的变化情况并生成可视化图表
"""

import sys
import os
sys.path.append('/home/user/IsaacLab/source/isaaclab/isaaclab/envs/mdp')

import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import json

# 导入vessels模块
import vessels

class VesselsVariableMonitor:
    def __init__(self):
        self.variables_history = defaultdict(list)
        self.step_count = 0
        self.start_time = time.time()
        
    def record_variables(self, vessel_system):
        """记录vessels系统中所有关键变量"""
        self.step_count += 1
        current_time = time.time() - self.start_time
        
        # 基本状态变量
        self.variables_history['eta'].append(vessel_system.eta.copy())
        self.variables_history['nu'].append(vessel_system.nu.copy())
        self.variables_history['eta_hat'].append(vessel_system.eta[[0,1,5]].copy())
        self.variables_history['nu_hat'].append(vessel_system.nu_hat.copy())
        self.variables_history['b_hat'].append(vessel_system.b_hat.copy())
        
        # 目标位置
        self.variables_history['target_position'].append(vessel_system.parameters_ref.copy())
        
        # 误差计算
        error = vessel_system.eta[[0,1,5]] - vessel_system.parameters_ref
        self.variables_history['error'].append(error.copy())
        self.variables_history['error_magnitude'].append(np.linalg.norm(error))
        
        # 速度大小
        self.variables_history['speed_magnitude'].append(np.linalg.norm(vessel_system.nu[[0,1,5]]))
        
        # 控制力
        if hasattr(vessel_system, 'u') and vessel_system.u is not None:
            self.variables_history['control_force'].append(vessel_system.u.copy())
            self.variables_history['control_force_magnitude'].append(np.linalg.norm(vessel_system.u))
        else:
            self.variables_history['control_force'].append(np.zeros(3))
            self.variables_history['control_force_magnitude'].append(0.0)
        
        # 控制参数
        self.variables_history['Kp'].append(vessel_system.Kp.copy())
        self.variables_history['Kd'].append(vessel_system.Kd.copy())
        
        # 系统矩阵
        self.variables_history['M'].append(vessel_system.M.copy())
        self.variables_history['D'].append(vessel_system.D.copy())
        self.variables_history['G'].append(vessel_system.G.copy())
        
        # 观测器相关变量
        if hasattr(vessel_system, 'x_hat'):
            self.variables_history['x_hat'].append(vessel_system.x_hat.copy())
        if hasattr(vessel_system, 'xi_hat'):
            self.variables_history['xi_hat'].append(vessel_system.xi_hat.copy())
        
        # 时间戳
        self.variables_history['time'].append(current_time)
        self.variables_history['step'].append(self.step_count)
        
        # 每100步打印一次关键信息
        if self.step_count % 100 == 0:
            print(f"第{self.step_count}步:")
            print(f"  位置: {vessel_system.eta[[0,1,5]]}")
            print(f"  目标: {vessel_system.parameters_ref}")
            print(f"  误差: {error}")
            print(f"  误差大小: {np.linalg.norm(error):.6f}")
            print(f"  速度大小: {np.linalg.norm(vessel_system.nu[[0,1,5]]):.6f}")
            if hasattr(vessel_system, 'u') and vessel_system.u is not None:
                print(f"  控制力: {vessel_system.u}")
                print(f"  控制力大小: {np.linalg.norm(vessel_system.u):.6f}")
            print(f"  b_hat: {vessel_system.b_hat}")
            print(f"  nu_hat: {vessel_system.nu_hat}")
            print("---")
    
    def analyze_variable_trends(self):
        """分析变量变化趋势"""
        print("\n=== 变量变化趋势分析 ===")
        
        # 分析误差变化
        if len(self.variables_history['error_magnitude']) > 10:
            error_trend = np.polyfit(range(len(self.variables_history['error_magnitude'])), 
                                   self.variables_history['error_magnitude'], 1)[0]
            print(f"误差变化趋势: {error_trend:.6f}")
            if error_trend > 0.001:
                print("⚠️  误差在增长")
            else:
                print("✅ 误差在下降或稳定")
        
        # 分析速度变化
        if len(self.variables_history['speed_magnitude']) > 10:
            speed_trend = np.polyfit(range(len(self.variables_history['speed_magnitude'])), 
                                   self.variables_history['speed_magnitude'], 1)[0]
            print(f"速度变化趋势: {speed_trend:.6f}")
            if speed_trend > 0.001:
                print("⚠️  速度在增长")
            else:
                print("✅ 速度在下降或稳定")
        
        # 分析控制力变化
        if len(self.variables_history['control_force_magnitude']) > 10:
            force_trend = np.polyfit(range(len(self.variables_history['control_force_magnitude'])), 
                                   self.variables_history['control_force_magnitude'], 1)[0]
            print(f"控制力变化趋势: {force_trend:.6f}")
            if force_trend > 1000:
                print("⚠️  控制力在快速增长")
            else:
                print("✅ 控制力相对稳定")
        
        # 分析b_hat变化
        if len(self.variables_history['b_hat']) > 10:
            b_hat_magnitudes = [np.linalg.norm(b) for b in self.variables_history['b_hat']]
            b_hat_trend = np.polyfit(range(len(b_hat_magnitudes)), b_hat_magnitudes, 1)[0]
            print(f"b_hat变化趋势: {b_hat_trend:.6f}")
            if b_hat_trend > 0.001:
                print("⚠️  b_hat在增长")
            else:
                print("✅ b_hat相对稳定")
        
        # 分析nu_hat变化
        if len(self.variables_history['nu_hat']) > 10:
            nu_hat_magnitudes = [np.linalg.norm(nu) for nu in self.variables_history['nu_hat']]
            nu_hat_trend = np.polyfit(range(len(nu_hat_magnitudes)), nu_hat_magnitudes, 1)[0]
            print(f"nu_hat变化趋势: {nu_hat_trend:.6f}")
            if nu_hat_trend > 0.001:
                print("⚠️  nu_hat在增长")
            else:
                print("✅ nu_hat相对稳定")
    
    def plot_variables(self, save_path="/home/user/IsaacLab/vessels_variables_analysis.png"):
        """绘制所有变量的变化图表"""
        if self.step_count == 0:
            print("没有数据可以绘制")
            return
        
        # 创建子图
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Vessels系统变量变化分析', fontsize=16)
        
        # 1. 位置变化
        if len(self.variables_history['eta_hat']) > 0:
            eta_hat_array = np.array(self.variables_history['eta_hat'])
            axes[0,0].plot(eta_hat_array[:, 0], label='X位置', color='red')
            axes[0,0].plot(eta_hat_array[:, 1], label='Y位置', color='green')
            axes[0,0].plot(eta_hat_array[:, 2], label='Z位置(偏航)', color='blue')
            axes[0,0].set_title('位置变化')
            axes[0,0].set_xlabel('步数')
            axes[0,0].set_ylabel('位置')
            axes[0,0].legend()
            axes[0,0].grid(True)
        
        # 2. 误差变化
        if len(self.variables_history['error']) > 0:
            error_array = np.array(self.variables_history['error'])
            axes[0,1].plot(error_array[:, 0], label='X误差', color='red')
            axes[0,1].plot(error_array[:, 1], label='Y误差', color='green')
            axes[0,1].plot(error_array[:, 2], label='Z误差', color='blue')
            axes[0,1].plot(self.variables_history['error_magnitude'], label='误差大小', color='black', linewidth=2)
            axes[0,1].set_title('误差变化')
            axes[0,1].set_xlabel('步数')
            axes[0,1].set_ylabel('误差')
            axes[0,1].legend()
            axes[0,1].grid(True)
        
        # 3. 速度变化
        if len(self.variables_history['nu_hat']) > 0:
            nu_hat_array = np.array(self.variables_history['nu_hat'])
            axes[0,2].plot(nu_hat_array[:, 0], label='X速度', color='red')
            axes[0,2].plot(nu_hat_array[:, 1], label='Y速度', color='green')
            axes[0,2].plot(nu_hat_array[:, 2], label='Z速度', color='blue')
            axes[0,2].plot(self.variables_history['speed_magnitude'], label='速度大小', color='black', linewidth=2)
            axes[0,2].set_title('速度变化')
            axes[0,2].set_xlabel('步数')
            axes[0,2].set_ylabel('速度')
            axes[0,2].legend()
            axes[0,2].grid(True)
        
        # 4. 控制力变化
        if len(self.variables_history['control_force']) > 0:
            control_force_array = np.array(self.variables_history['control_force'])
            axes[1,0].plot(control_force_array[:, 0], label='X控制力', color='red')
            axes[1,0].plot(control_force_array[:, 1], label='Y控制力', color='green')
            axes[1,0].plot(control_force_array[:, 2], label='Z控制力', color='blue')
            axes[1,0].plot(self.variables_history['control_force_magnitude'], label='控制力大小', color='black', linewidth=2)
            axes[1,0].set_title('控制力变化')
            axes[1,0].set_xlabel('步数')
            axes[1,0].set_ylabel('控制力')
            axes[1,0].legend()
            axes[1,0].grid(True)
            axes[1,0].set_yscale('log')  # 使用对数坐标
        
        # 5. b_hat变化
        if len(self.variables_history['b_hat']) > 0:
            b_hat_array = np.array(self.variables_history['b_hat'])
            axes[1,1].plot(b_hat_array[:, 0], label='b_hat[0]', color='red')
            axes[1,1].plot(b_hat_array[:, 1], label='b_hat[1]', color='green')
            axes[1,1].plot(b_hat_array[:, 2], label='b_hat[2]', color='blue')
            b_hat_magnitudes = [np.linalg.norm(b) for b in self.variables_history['b_hat']]
            axes[1,1].plot(b_hat_magnitudes, label='b_hat大小', color='black', linewidth=2)
            axes[1,1].set_title('b_hat变化')
            axes[1,1].set_xlabel('步数')
            axes[1,1].set_ylabel('b_hat')
            axes[1,1].legend()
            axes[1,1].grid(True)
        
        # 6. nu_hat变化
        if len(self.variables_history['nu_hat']) > 0:
            nu_hat_array = np.array(self.variables_history['nu_hat'])
            axes[1,2].plot(nu_hat_array[:, 0], label='nu_hat[0]', color='red')
            axes[1,2].plot(nu_hat_array[:, 1], label='nu_hat[1]', color='green')
            axes[1,2].plot(nu_hat_array[:, 2], label='nu_hat[2]', color='blue')
            nu_hat_magnitudes = [np.linalg.norm(nu) for nu in self.variables_history['nu_hat']]
            axes[1,2].plot(nu_hat_magnitudes, label='nu_hat大小', color='black', linewidth=2)
            axes[1,2].set_title('nu_hat变化')
            axes[1,2].set_xlabel('步数')
            axes[1,2].set_ylabel('nu_hat')
            axes[1,2].legend()
            axes[1,2].grid(True)
        
        # 7. 系统矩阵M的变化
        if len(self.variables_history['M']) > 0:
            M_diag = []
            for M in self.variables_history['M']:
                M_diag.append(np.diag(M))
            M_diag_array = np.array(M_diag)
            for i in range(6):
                axes[2,0].plot(M_diag_array[:, i], label=f'M[{i},{i}]')
            axes[2,0].set_title('质量矩阵M对角线元素变化')
            axes[2,0].set_xlabel('步数')
            axes[2,0].set_ylabel('M值')
            axes[2,0].legend()
            axes[2,0].grid(True)
            axes[2,0].set_yscale('log')
        
        # 8. 系统矩阵D的变化
        if len(self.variables_history['D']) > 0:
            D_diag = []
            for D in self.variables_history['D']:
                D_diag.append(np.diag(D))
            D_diag_array = np.array(D_diag)
            for i in range(6):
                axes[2,1].plot(D_diag_array[:, i], label=f'D[{i},{i}]')
            axes[2,1].set_title('阻尼矩阵D对角线元素变化')
            axes[2,1].set_xlabel('步数')
            axes[2,1].set_ylabel('D值')
            axes[2,1].legend()
            axes[2,1].grid(True)
            axes[2,1].set_yscale('log')
        
        # 9. 系统矩阵G的变化
        if len(self.variables_history['G']) > 0:
            G_diag = []
            for G in self.variables_history['G']:
                G_diag.append(np.diag(G))
            G_diag_array = np.array(G_diag)
            for i in range(6):
                axes[2,2].plot(G_diag_array[:, i], label=f'G[{i},{i}]')
            axes[2,2].set_title('重力矩阵G对角线元素变化')
            axes[2,2].set_xlabel('步数')
            axes[2,2].set_ylabel('G值')
            axes[2,2].legend()
            axes[2,2].grid(True)
            axes[2,2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"变量变化图表已保存到: {save_path}")
        plt.show()
    
    def save_data(self, save_path="/home/user/IsaacLab/vessels_variables_data.json"):
        """保存变量数据到JSON文件"""
        # 转换numpy数组为列表以便JSON序列化
        data_to_save = {}
        for key, values in self.variables_history.items():
            if isinstance(values[0], np.ndarray):
                data_to_save[key] = [v.tolist() for v in values]
            else:
                data_to_save[key] = values
        
        with open(save_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"变量数据已保存到: {save_path}")

def run_monitoring_simulation():
    """运行监控仿真"""
    print("开始监控vessels系统变量变化...")
    
    # 创建监控器
    monitor = VesselsVariableMonitor()
    
    # 创建vessels系统
    vessel = vessels.VesselControlSystem(target_position=[1, 1, 3.1415926])
    
    print(f"目标位置: {vessel.parameters_ref}")
    print(f"初始位置: {vessel.eta[[0,1,5]]}")
    
    # 运行仿真并记录变量
    max_steps = 2000  # 运行2000步
    print(f"开始运行{max_steps}步仿真...")
    
    for i in range(max_steps):
        current_eta = vessel.eta.copy()
        current_nu = vessel.nu.copy()
        acc, eta_dot = vessel.step(current_eta, current_nu, i * 0.02)
        vessel.eta = vessel.eta + eta_dot * 0.02
        vessel.nu = vessel.nu + acc * 0.02
        
        # 记录变量
        monitor.record_variables(vessel)
        
        # 检查是否发散
        if i > 100:
            error_mag = np.linalg.norm(vessel.eta[[0,1,5]] - vessel.parameters_ref)
            if error_mag > 50:  # 如果误差过大，认为发散
                print(f"❌ 第{i+1}步: 系统发散，误差={error_mag:.6f}")
                break
    
    print(f"\n仿真完成，共运行{monitor.step_count}步")
    
    # 分析变量变化趋势
    monitor.analyze_variable_trends()
    
    # 绘制变量变化图表
    monitor.plot_variables()
    
    # 保存数据
    monitor.save_data()
    
    return monitor

if __name__ == "__main__":
    monitor = run_monitoring_simulation()
