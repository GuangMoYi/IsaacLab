# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# GMY 
import math
def move_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    position_range: dict[str, tuple[float, float]] = {
        "x": (-1e6, 1e6), 
        "y": (-1e6, 1e6), 
        "z": (-1e6, 1e6), 
        "roll": (-math.pi, math.pi), 
        "pitch": (-0.5 * math.pi, 0.5 * math.pi), 
        "yaw": (-math.pi, math.pi) },  # 添加位置范围
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    overwrite_velocity: bool =  False,  # 添加控制参数
):
    """  
        函数作用： 在设置的速度范围内随机抽取某速度以控制刚体运动，并限制刚体运动幅度
        输入参数：
            env: ManagerBasedEnv, 环境实例
            env_ids: torch.Tensor, 环境ID
            velocity_range: dict[str, tuple[float, float]]  设置速度选取范围{"x":(min, max) , "y", "z", "roll", "pitch", "yaw"}
            position_range: dict[str, tuple[float, float]]  设置自由度幅度{"x":(min, max), "y", "z", "roll", "pitch", "yaw"}
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  作用刚体的名称
            overwrite_velocity: bool =  False,  是否叠加加速度： False 叠加， True 不叠加
    """
    
    range_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    # 读取当前速度
    vel_w = asset.data.root_vel_w[env_ids]
    # 采样随机速度
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in range_keys]
    ranges = torch.tensor(range_list, device=asset.device)
    
    
    #-------------------------------------------------------------------------------------------------------# 
    if not hasattr(env, '_initial_platform_rot'):                           # 初始四元数 [w, x, y, z]
        setattr(env, '_initial_platform_rot', asset.data.root_quat_w.clone())
    initial_quat = getattr(env, '_initial_platform_rot')[env_ids]
    if not hasattr(env, '_initial_platform_pos'):                           # 初始位置 [x, y, z]
        setattr(env, '_initial_platform_pos', asset.data.root_pos_w.clone()) 
    initial_pos = getattr(env, '_initial_platform_pos')[env_ids]
    time = 0.25 * env._sim_step_counter * env.physics_dt           
    
    current_quat = asset.data.root_quat_w[env_ids]                          # 读取当前四元数
    
        # 计算相对旋转 (current_quat * initial_quat^-1)
    q_rel = math_utils.quat_mul(current_quat, math_utils.quat_conjugate(initial_quat.clone().detach()))  
        # 将相对旋转转换为旋转角度（弧度）        
    rot_angles = torch.stack(math_utils.euler_xyz_from_quat(q_rel), dim=1)  # 读取旋转角度 [roll, pitch, yaw]
    rot_angles = (rot_angles + math.pi) % (2 * math.pi) - math.pi           # 归一化到 [-pi, pi]： 这是因为有些角度2*pi没法处理
    
    current_pos = asset.data.root_pos_w[env_ids]                            # 读取当前位置
    relative_pos = current_pos - initial_pos

    # 拼接位置和角度
    pose = torch.cat([relative_pos, rot_angles], dim=1)
    current_pose = torch.cat([current_pos, torch.stack(math_utils.euler_xyz_from_quat(current_quat), dim=1)], dim=1)


    # 读取位置和角度范围
    position_range_list = [position_range.get(key, (-1e6, 1e6)) for key in range_keys]
    position_range_list = torch.tensor(position_range_list, device=asset.device)

    # 读取平台的位置和角度
    platform = env.scene["platform"]
    platform_pos = platform.data.root_pos_w[env_ids]
    platform_quat = platform.data.root_quat_w[env_ids]
    platform_rot = torch.stack(math_utils.euler_xyz_from_quat(platform_quat), dim=1)
    platform_pose = torch.cat([platform_pos, platform_rot], dim=1) 

    # 计算哪些维度的物体超出范围(与平台的相对位置，注意paltfrom_pose的位置是平台中心的)
    platform_pose_judge = platform_pose.clone()
    platform_pose_judge[:, 2] += 0.5 * platform.cfg.spawn.size[2] 
    too_high = current_pose - platform_pose_judge  >= position_range_list[:, 1]                            # 超过最大值
    too_low = current_pose - platform_pose_judge <= position_range_list[:, 0]                             # 低于最小值

    # print("[INFO] pose: ", pose)
    # print("[INFO] platform_pose_judge: ", platform_pose_judge)
    # print("[INFO] 差值: ", pose - platform_pose_judge)

                                                                            # pose应为[env_num,6]
                                                                            # position_range_list应为[6,2]
                                                                            # too_high应为[env_num,6]
                                                                            # ranges应为[6,2]
    # 根据 overwrite_velocity 决定是否叠加速度
    t = torch.tensor(time, device=asset.device) # t 是 float，需要转为 tensor
    A = ranges[:, 0]    # 振幅 (N,)
    phi = ranges[:, 1]  # 相位 (N,)
    T_tmie = 5        # 周期 (s)
    omega = 2 * math.pi / T_tmie  # 角速度 (rad/s)  # 0.05 是周期，单位为秒

    env_num, dof = vel_w.shape      # N 是环境数量，dof 是自由度数量
    A = A.unsqueeze(0).expand(env_num, -1)         # shape [1, dof]
    phi = phi.unsqueeze(0).expand(env_num, -1)     # shape [1, dof]
    
    if overwrite_velocity:
        vel_w = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
        # vel_w = A * torch.sin(omega * t + phi)
        
    else:
        vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
        # vel_w += A * torch.sin(omega * t + phi)

    # # 遍历每个自由度维度，处理超范围的速度
    # for i in range(vel_w.shape[1]):  
    #     # 超出最大范围，且速度方向向外（正）
    #     mask_high = too_high[:, i] & (vel_w[:, i] > 0)
    #     vel_w[:, i] = torch.where(mask_high, torch.zeros_like(vel_w[:, i]), vel_w[:, i])

    #     # 低于最小范围，且速度方向向外（负）
    #     mask_low = too_low[:, i] & (vel_w[:, i] < 0)
    #     vel_w[:, i] = torch.where(mask_low, torch.zeros_like(vel_w[:, i]), vel_w[:, i])

    # mask 部分原地使用广播就可以，不需要 for 循环
    mask_high = too_high # & (vel_w > 0)
    vel_w = torch.where(mask_high, -ranges[:, 1].unsqueeze(0).expand_as(vel_w), vel_w)

    mask_low = too_low # & (vel_w < 0)
    vel_w = torch.where(mask_low, -ranges[:, 0].unsqueeze(0).expand_as(vel_w), vel_w)

    # print(f"current_pose: {current_pose}")
    # print(f"platform_pose_judge: {platform_pose_judge}")
    # print(f"差值: {current_pose - platform_pose_judge}")
    # print(f"position_range_list: {position_range_list}")
    # print(f"too_high: {too_high}")
    # print(f"too_low: {too_low}")
    # print(f"vel_w: {vel_w}")

    
    #-------------------------------------------------------------------------------------------------------#

    # 应用速度到物理仿真
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

from isaaclab.envs.mdp.vessels import frigate, semisub, supply, VesselControlSystem
def move_acceleration(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
):
    import numpy as np
    import os
    
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    

    # 初始化位姿参考
    if not hasattr(env, '_initial_platform_rot'):
        setattr(env, '_initial_platform_rot', asset.data.root_quat_w.clone())
    initial_quat = getattr(env, '_initial_platform_rot')[env_ids]
    if not hasattr(env, '_initial_platform_pos'):
        setattr(env, '_initial_platform_pos', asset.data.root_pos_w.clone())
    initial_pos = getattr(env, '_initial_platform_pos')[env_ids]

    dt = 0.02
    time_me = (0.25 * env._sim_step_counter)
    
    # 调试信息：检查时间步长
    if env_ids[0] == 0 and hasattr(env, '_debug_counter'):
        env._debug_counter += 1
    else:
        env._debug_counter = 1
    # print("[INFO] 循环函数次数:", time_me)

    current_quat = asset.data.root_quat_w[env_ids]
    q_rel = math_utils.quat_mul(current_quat, math_utils.quat_conjugate(initial_quat.clone().detach()))
    rot_angles = torch.stack(math_utils.euler_zyx_from_quat(current_quat), dim=1)
    # 关键：归一化角度到[-pi, pi]，防止角度累积导致发散
    rot_angles = math_utils.wrap_to_pi(rot_angles)

    current_pos = asset.data.root_pos_w[env_ids]
    relative_pos = current_pos - initial_pos
    pose = torch.cat([relative_pos, rot_angles], dim=1)

    current_pose = torch.cat([current_pos, rot_angles], dim=1)

    # 先计算速度数据
    nu_lin = asset.data.root_lin_vel_b[env_ids]
    nu_ang = asset.data.root_ang_vel_b[env_ids] 
    nu = torch.cat([nu_lin, nu_ang], dim=-1)
    
    # ========================================================平台运动记录和读取功能===========================================
    # 初始化平台运动记录和读取相关的属性
    if not hasattr(env, '_platform_motion_recorded'):
        env._platform_motion_recorded = False
    if not hasattr(env, '_platform_motion_data'):
        env._platform_motion_data = None
    if not hasattr(env, '_platform_motion_file_path'):
        save_dir = "/home/user/IsaacLab/comparison_data"
        os.makedirs(save_dir, exist_ok=True)
        env._platform_motion_file_path = os.path.join(save_dir, "platform_motion_1000.npz")
    if not hasattr(env, '_platform_motion_step_index'):
        env._platform_motion_step_index = 0
    if not hasattr(env, '_platform_motion_history'):
        env._platform_motion_history = {'eta': [], 'nu': []}  # 记录前1000步的位置和速度
    
    # 获取当前步数（使用第一个环境的步数）
    current_step = int(time_me * 4)  # time_me = 0.25 * env._sim_step_counter，所以步数 = time_me * 4
    
    # 检查平台运动文件是否存在
    if os.path.exists(env._platform_motion_file_path) and env._platform_motion_data is None:
        # 如果文件存在且还没有加载，则加载文件
        print(f"[INFO] 发现平台运动文件，正在加载: {env._platform_motion_file_path}")
        loaded_data = np.load(env._platform_motion_file_path)
        env._platform_motion_data = {
            'eta': loaded_data['eta'],  # [1000, 6] 位置和角度
            'nu': loaded_data['nu']     # [1000, 6] 速度
        }
        print(f"[INFO] 平台运动数据已加载，共 {len(env._platform_motion_data['eta'])} 步")
        env._platform_motion_recorded = True
    
    # 如果文件不存在且还没有记录，则记录前1000步的平台运动
    if not env._platform_motion_recorded and current_step < 1000:
        # 记录当前步的平台运动（位置和速度）
        pose_np = pose[0].detach().cpu().numpy()  # [6] 相对位置和角度
        nu_np = nu[0].detach().cpu().numpy()      # [6] 体坐标系速度
        
        # 关键修复：确保角度归一化到[-pi, pi]，防止角度累积
        import math
        for i in range(3, 6):  # 角度索引：3=roll, 4=pitch, 5=yaw
            angle = pose_np[i]
            # 归一化角度到[-pi, pi]
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle < -math.pi:
                angle += 2 * math.pi
            pose_np[i] = angle
        
        # 关键修复：确保第0步的初始状态为0（双重保险）
        if current_step == 0:
            pose_np = np.zeros(6, dtype=pose_np.dtype)  # 强制初始位置和角度为0
            nu_np = np.zeros(6, dtype=nu_np.dtype)      # 强制初始速度为0
            print(f"[INFO] 强制初始状态为0，确保完美的周期循环")
        
        env._platform_motion_history['eta'].append(pose_np.copy())
        env._platform_motion_history['nu'].append(nu_np.copy())
        
        if current_step == 10000:  # 第1000步（索引999）时保存文件
            print(f"[INFO] 前1000步平台运动记录完成，正在保存到文件...")
            eta_array = np.array(env._platform_motion_history['eta'])  # [1000, 6]
            nu_array = np.array(env._platform_motion_history['nu'])     # [1000, 6]
            
            # 关键修复：确保初始状态为0
            eta_array[0] = np.zeros(6)
            nu_array[0] = np.zeros(6)
            
            # 关键修复：确保所有角度都归一化到[-pi, pi]
            for i in range(len(eta_array)):
                for j in range(3, 6):  # 角度索引
                    angle = eta_array[i, j]
                    while angle > math.pi:
                        angle -= 2 * math.pi
                    while angle < -math.pi:
                        angle += 2 * math.pi
                    eta_array[i, j] = angle
            
            # 关键修复：检查最后一步的状态，确保循环时状态连续
            # 注意：不再强制平滑过渡，而是检查运动是否自然回到0
            # 如果运动是周期性的，最后一步应该自然接近0
            final_eta = eta_array[-1].copy()
            final_nu = nu_array[-1].copy()
            
            # 检查最后一步是否接近0（允许小的数值误差）
            final_eta_magnitude = np.linalg.norm(final_eta)
            final_nu_magnitude = np.linalg.norm(final_nu)
            
            if final_eta_magnitude > 0.01 or final_nu_magnitude > 0.01:
                # 如果最后一步离0较远，说明运动不是完美的周期，需要平滑过渡
                print(f"[INFO] 检测到最后一步状态离0较远，将在最后几步平滑过渡到0")
                print(f"  最后一步状态: eta={final_eta}, nu={final_nu}")
                print(f"  位置范数: {final_eta_magnitude:.6f}, 速度范数: {final_nu_magnitude:.6f}")
                
                # 使用更少的平滑步数，避免破坏周期性
                smooth_steps = min(50, len(eta_array) // 10)  # 最多50步或总步数的10%
                
                # 在最后smooth_steps步中，逐渐将位置和速度平滑到0
                for i in range(max(0, len(eta_array) - smooth_steps), len(eta_array)):
                    # 计算平滑系数：从0到1（使用平滑函数，避免突然变化）
                    alpha = (i - (len(eta_array) - smooth_steps)) / smooth_steps
                    # 使用平滑函数（smoothstep）而不是线性插值
                    alpha_smooth = alpha * alpha * (3 - 2 * alpha)
                    # 线性插值：从原始值平滑到0
                    eta_array[i] = (1 - alpha_smooth) * eta_array[i] + alpha_smooth * np.zeros(6)
                    nu_array[i] = (1 - alpha_smooth) * nu_array[i] + alpha_smooth * np.zeros(6)
                
                # 确保最后一步完全为0
                eta_array[-1] = np.zeros(6)
                nu_array[-1] = np.zeros(6)
                print(f"[INFO] 平滑过渡完成，最后一步状态: eta={eta_array[-1]}, nu={nu_array[-1]}")
            else:
                # 如果最后一步已经接近0，直接设为0（双重保险）
                eta_array[-1] = np.zeros(6)
                nu_array[-1] = np.zeros(6)
                print(f"[INFO] 最后一步已接近0，直接设为0（位置范数: {final_eta_magnitude:.6f}, 速度范数: {final_nu_magnitude:.6f}）")
            
            np.savez(env._platform_motion_file_path,
                     eta=eta_array,
                     nu=nu_array)
            # print(f"[INFO] 平台运动数据已保存到: {env._platform_motion_file_path}")
            # print(f"[INFO] 初始状态: eta[0]={eta_array[0]}, nu[0]={nu_array[0]}")
            # print(f"[INFO] 最终状态: eta[999]={eta_array[-1]}, nu[999]={nu_array[-1]}")
            env._platform_motion_recorded = True
            # 加载保存的数据以便后续使用
            env._platform_motion_data = {
                'eta': eta_array,
                'nu': nu_array
            }
    
    # 如果已经记录完成或文件已加载，则从文件读取平台运动（实现直接循环）
    if env._platform_motion_recorded and env._platform_motion_data is not None:
        # 关键改进：直接循环读取，不需要反向
        # 前提：确保第0步和第1000步（最后一步）的状态都是0
        # 这样从第1000步循环回第0步时，状态完全连续，不会跳跃
        #
        # 实现方式：
        # - 在记录数据时，最后几步平滑过渡到0，确保最后一步状态为0
        # - 在读取数据时，直接循环读取：0->1->...->999->0->1->...
        # - 这样每个周期（1000步）后，平台都会回到初始状态，可以无限循环
        
        total_steps = len(env._platform_motion_data['eta'])
        
        # 关键修复：检测是否从第999步过渡到第0步（循环边界）
        # 在更新索引之前检查，这样可以准确判断循环边界
        previous_step_index = env._platform_motion_step_index
        is_cycle_boundary = (previous_step_index == total_steps - 1)  # 从第999步过渡到第0步
        
        # 直接循环读取：计算当前步数索引（0-999循环）
        step_index = env._platform_motion_step_index % total_steps
        
        # 关键修复：在循环边界时，强制使用初始状态（0状态），确保完美循环
        # 这样可以消除物理引擎在1000步运行中产生的累积误差
        if is_cycle_boundary:
            # 循环边界：从第999步过渡到第0步，强制使用初始状态（相对位置和速度为0）
            next_eta_from_file = np.zeros(6, dtype=np.float64)
            next_nu_from_file = np.zeros(6, dtype=np.float64)
            
            # 验证文件中的初始状态确实为0（用于调试）
            initial_eta = env._platform_motion_data['eta'][0].copy()
            initial_nu = env._platform_motion_data['nu'][0].copy()
            if not np.allclose(initial_eta, np.zeros(6), atol=1e-4):
                print(f"[WARNING] 文件中的初始状态不为0！eta[0]={initial_eta}")
            if not np.allclose(initial_nu, np.zeros(6), atol=1e-4):
                print(f"[WARNING] 文件中的初始速度不为0！nu[0]={initial_nu}")
        else:
            # 非循环边界：正常从文件读取数据
            next_eta_from_file = env._platform_motion_data['eta'][step_index].copy()
            next_nu_from_file = env._platform_motion_data['nu'][step_index].copy()
            
            # 关键修复：确保角度归一化到[-pi, pi]，防止角度累积导致发散
            import math
            for i in range(3, 6):  # 角度索引：3=roll, 4=pitch, 5=yaw
                angle = next_eta_from_file[i]
                # 归一化角度到[-pi, pi]
                while angle > math.pi:
                    angle -= 2 * math.pi
                while angle < -math.pi:
                    angle += 2 * math.pi
                next_eta_from_file[i] = angle
        
        # 更新索引，循环读取（每1000步一个完整周期）
        # 这样确保每个完整周期后，平台都回到初始状态，可以无限循环
        env._platform_motion_step_index = (env._platform_motion_step_index + 1) % total_steps
        
        # 将读取的数据转换为torch张量
        next_eta_torch = torch.from_numpy(next_eta_from_file).to(
            dtype=current_quat.dtype,
            device=current_quat.device
        ).unsqueeze(0).expand(len(env_ids), -1)  # [num_envs, 6]
        
        # 计算绝对位置（加上初始位置偏移）
        next_pos_relative = next_eta_torch[:, :3].clone()  # [num_envs, 3] 相对位置
        # z坐标正常使用，不进行强制限制
        next_pos_world = next_pos_relative + initial_pos  # [num_envs, 3]
        
        # 处理姿态：next_eta[3:6] 包含欧拉角，顺序是 (roll, pitch, yaw) - ZYX顺序
        next_roll = next_eta_torch[:, 3]
        next_pitch = next_eta_torch[:, 4]
        next_yaw = next_eta_torch[:, 5]
        
        # ZYX 顺序的欧拉角转换为四元数
        cy = torch.cos(next_yaw * 0.5)
        sy = torch.sin(next_yaw * 0.5)
        cp = torch.cos(next_pitch * 0.5)
        sp = torch.sin(next_pitch * 0.5)
        cr = torch.cos(next_roll * 0.5)
        sr = torch.sin(next_roll * 0.5)
        
        # ZYX 顺序的四元数转换
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        next_quat = torch.stack([qw, qx, qy, qz], dim=-1)  # [num_envs, 4]
        
        # 组合位置和姿态
        next_root_pose = torch.cat([next_pos_world, next_quat], dim=-1)  # [num_envs, 7]
        
        # 将速度从体坐标系转换为世界坐标系
        # 注意：next_nu_from_file 已经是相反数了
        next_nu_torch = torch.from_numpy(next_nu_from_file).to(
            dtype=current_quat.dtype,
            device=current_quat.device
        ).unsqueeze(0).expand(len(env_ids), -1)  # [num_envs, 6]
        
        # 使用四元数转换体坐标系速度到世界坐标系
        lin_vel_body = next_nu_torch[:, :3]  # [num_envs, 3] 体坐标系线速度
        lin_vel_world = math_utils.quat_apply(next_quat, lin_vel_body)  # [num_envs, 3] 世界坐标系线速度
        
        ang_vel_body = next_nu_torch[:, 3:6]  # [num_envs, 3] 体坐标系角速度
        ang_vel_world = math_utils.quat_apply(next_quat, ang_vel_body)  # [num_envs, 3] 世界坐标系角速度
        
        # 组合线速度和角速度（都是世界坐标系）
        next_root_velocity = torch.cat([lin_vel_world, ang_vel_world], dim=-1)  # [num_envs, 6]
        
        # 关键修复：在循环边界时，强制重置平台物理状态
        # 这样可以消除物理引擎在1000步运行中产生的累积误差，确保完美循环
        if is_cycle_boundary:
            # 循环边界：强制使用初始位置和姿态（相对于initial_pos和initial_quat）
            next_pos_world = initial_pos.clone()
            next_quat = initial_quat.clone()
            next_root_pose = torch.cat([next_pos_world, next_quat], dim=-1)  # [num_envs, 7]
            # 强制速度为0
            next_root_velocity = torch.zeros(len(env_ids), 6, dtype=next_root_velocity.dtype, device=next_root_velocity.device)
            # 只在第一个环境且每10个周期打印一次日志
            if env_ids[0] == 0:
                cycle_count = previous_step_index // total_steps if previous_step_index >= total_steps else 0
                if cycle_count % 10 == 0:  # 每10个周期打印一次
                    print(f"[INFO] 循环边界重置：强制平台回到初始状态，消除累积误差（周期: {cycle_count}, 从步{previous_step_index}到步0）")
        
        # 直接设置平台位置和姿态（所有平台完全同步）
        asset.write_root_pose_to_sim(next_root_pose, env_ids=env_ids)
        # 同时设置速度（所有平台完全同步）
        asset.write_root_velocity_to_sim(next_root_velocity, env_ids=env_ids)
        
        # 从文件读取后直接返回，不再执行后续的控制系统代码
        return
    # ========================================================平台运动记录和读取功能===========================================

    # ========================================================原有不同环境不同平台代码（已注释，改用共享系统）===========================================
    # # 为每个环境创建独立的VesselControlSystem实例
    # if not hasattr(env, '_vehicle_dict'):
    #     env._vehicle_dict = {}
    # ... (已注释的独立系统代码)
    # ========================================================原有不同环境不同平台代码===========================================

    # ========================================================所有环境同一个平台代码===========================================
    # 性能优化：所有环境共用一个控制平台系统，大幅提升多环境性能
    # 物理上：每个环境都有独立的物理平台刚体（1024个物理平台）
    # 控制上：所有平台都使用同一个VesselControlSystem实例，执行相同的运动轨迹
    # 所有平台的位置、速度、加速度完全同步

    # 初始化对比数据存储（保留这个，因为优化后的代码也需要）
    if not hasattr(env, '_comparison_data'):
        env._comparison_data = {}
    
    # 关键优化1：创建全局共享的控制平台系统（只创建一次）
    if not hasattr(env, '_shared_platform_system'):
        # 使用第一个环境的初始状态创建共享平台系统
        initial_pose = pose[0].detach().cpu().numpy()  # 使用第一个环境的相对位置
        initial_nu = nu[0].detach().cpu().numpy()      # 使用第一个环境的速度（体坐标系）
        target_position = [0, 0, 0 * np.pi]        # 期望位置
        
        # 关键修复：在记录阶段，使用固定相位确保平台运动是周期性的
        # 如果正在记录平台运动（前1000步），使用固定相位；否则使用随机相位
        is_recording = not env._platform_motion_recorded and current_step < 1000
        use_fixed_phase = is_recording  # 记录时使用固定相位，确保周期性
        
        env._shared_platform_system = VesselControlSystem(
                target_position=target_position,
            initial_eta=initial_pose,
            initial_nu=initial_nu,
                dt=dt,
                use_fixed_phase=use_fixed_phase  # 传递固定相位标志
            )
        if use_fixed_phase:
            print(f"[INFO] 创建全局共享控制平台系统（记录模式：使用固定相位确保周期性），目标位置: {target_position}")
        else:
            print(f"[INFO] 创建全局共享控制平台系统，目标位置: {target_position}")
    
    # 关键优化2：批量处理所有环境，避免循环
    # 将torch张量转换为numpy数组进行批量计算
    pose_np = pose.detach().cpu().numpy()  # [num_envs, 6] 相对位置
    nu_np = nu.detach().cpu().numpy()      # [num_envs, 6] 体坐标系速度
    current_time = time_me * dt
    
    # 关键优化3：使用共享控制策略
    # 所有环境使用相同的控制平台系统，执行同步运动
    shared_system = env._shared_platform_system
    
    # 预分配输出数组
    num_envs = len(env_ids)
    next_eta_batch = np.zeros((num_envs, 6))
    eta_dot_batch = np.zeros((num_envs, 6))
    
    # 关键优化4：真正的共享控制 - 所有平台使用相同的控制指令
    # 只使用第一个环境的状态来计算控制指令（所有平台同步运动）
    # 闭环控制：从物理引擎读取实际状态作为反馈，计算期望状态后设置回去
    # 这样可以处理物理引擎的数值误差、碰撞等因素导致的偏差
    
    # 关键：使用从物理引擎读取的实际状态（pose_np[0], nu_np[0]）作为输入
    # 这是闭环控制的关键：用实际状态作为反馈，而不是期望状态
    # 第一次调用时初始化系统状态
    if not hasattr(shared_system, '_state_initialized'):
        shared_system.eta[:] = pose_np[0]  # 第一次调用时初始化位置
        shared_system.nu[:] = nu_np[0]     # 第一次调用时初始化速度
        shared_system._state_initialized = True
    
    # 关键修复：确保角度归一化到[-pi, pi]，防止角度累积导致发散
    # 在传递给控制系统之前，对角度部分进行归一化
    current_eta_normalized = pose_np[0].copy()
    current_eta_normalized[3:6] = np.arctan2(
        np.sin(current_eta_normalized[3:6]),
        np.cos(current_eta_normalized[3:6])
    )
    
    # 只计算一次控制指令
    # 注意：vessels.py中的step函数返回next_eta和eta_dot，而不是加速度
    # 使用从物理引擎读取的实际状态（归一化后的角度）作为输入
    # 这是闭环控制：实际状态 -> 控制系统 -> 期望状态 -> 物理引擎 -> 实际状态（下一帧）
    next_eta, eta_dot = shared_system.step(current_eta_normalized, nu_np[0], current_time)
    
    # 关键修复：确保返回的角度也归一化到[-pi, pi]
    next_eta[3:6] = np.arctan2(np.sin(next_eta[3:6]), np.cos(next_eta[3:6]))
    
    # 将相同的控制指令应用到所有环境（所有平台同步运动）
    for i in range(num_envs):
        next_eta_batch[i] = next_eta
        eta_dot_batch[i] = eta_dot
    
    # 关键优化5：将numpy结果转换为torch张量，并转换为IsaacLab格式
    # next_eta: [x, y, z, roll, pitch, yaw] (相对位置，相对于initial_pos)
    # 需要转换为IsaacLab格式：[位置(3), 四元数(4)]
    
    next_eta_torch = torch.from_numpy(next_eta_batch).to(
        dtype=current_quat.dtype, 
        device=current_quat.device
    )
    
    # 计算绝对位置（加上初始位置偏移）
    # next_eta的前3个元素是相对位置 [x, y, z]，相对于每个环境的initial_pos
    # 注意：所有环境使用相同的next_eta（相对位置），但每个环境加上它自己的initial_pos
    # z坐标正常使用，不进行强制限制
    next_pos_relative = next_eta_torch[:, :3].clone()  # [num_envs, 3] 相对位置
    next_pos_world = next_pos_relative + initial_pos  # [num_envs, 3] 每个环境加上它自己的初始位置
    
    # 处理姿态：next_eta[3:6] 包含欧拉角，顺序是 (roll, pitch, yaw) - ZYX顺序
    # 转换为四元数（参考events_acc.py，但events_acc.py使用的是加速度，这里使用位置）
    next_roll = next_eta_torch[:, 3]
    next_pitch = next_eta_torch[:, 4]
    next_yaw = next_eta_torch[:, 5]
    
    # ZYX 顺序的欧拉角转换为四元数
    # ZYX 顺序：先绕 Z 轴旋转 yaw，然后绕 Y 轴旋转 pitch，最后绕 X 轴旋转 roll
    # 四元数乘法顺序：q_roll * q_pitch * q_yaw
    cy = torch.cos(next_yaw * 0.5)
    sy = torch.sin(next_yaw * 0.5)
    cp = torch.cos(next_pitch * 0.5)
    sp = torch.sin(next_pitch * 0.5)
    cr = torch.cos(next_roll * 0.5)
    sr = torch.sin(next_roll * 0.5)
    
    # ZYX 顺序的四元数转换
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    next_quat = torch.stack([qw, qx, qy, qz], dim=-1)  # [num_envs, 4]
    
    # 组合位置和姿态
    next_root_pose = torch.cat([next_pos_world, next_quat], dim=-1)  # [num_envs, 7]
    
    # 关键优化6：保持原有的对比数据功能（仅对第一个环境）
    # 确保第一个环境的对比数据被初始化
    if 0 in env_ids.tolist():
        if 0 not in env._comparison_data:
            print(f"[INFO] 初始化环境0的对比数据")
            env._comparison_data[0] = {
                'isaaclab_eta_history': [],
                'isaaclab_nu_history': [],
                'calculated_eta_history': [],
                'calculated_nu_history': [],
                'calculated_eta': np.zeros(6),  # 从0开始积分
                'calculated_nu': np.zeros(6),   # 从0开始积分
                'step_count': 0
            }
    
    if 0 in env_ids.tolist() and 0 in env._comparison_data:
        comp_data = env._comparison_data[0]
        comp_data['step_count'] += 1
        
        # 存储IsaacLab的真实输出
        isaaclab_eta = pose[0].detach().cpu().numpy()
        isaaclab_nu = nu[0].detach().cpu().numpy()
        comp_data['isaaclab_eta_history'].append(isaaclab_eta.copy())
        comp_data['isaaclab_nu_history'].append(isaaclab_nu.copy())
        
        # 积分计算对比（现在使用期望位置进行对比）
        if comp_data['step_count'] == 1:
            comp_data['calculated_eta'] = isaaclab_eta.copy()
            comp_data['calculated_nu'] = isaaclab_nu.copy()
        else:
            # 使用期望位置进行对比（直接设置位置，所以直接使用期望位置）
            next_eta_np = next_eta_batch[0]
            
            # 直接使用期望位置（因为现在我们是直接设置位置）
            comp_data['calculated_eta'] = next_eta_np.copy()
            comp_data['calculated_nu'] = shared_system.nu.copy()  # 使用系统内部速度
            
            # 角度包装
            comp_data['calculated_eta'][3:] = np.arctan2(
                np.sin(comp_data['calculated_eta'][3:]), 
                np.cos(comp_data['calculated_eta'][3:])
            )
        
        comp_data['calculated_eta_history'].append(comp_data['calculated_eta'].copy())
        comp_data['calculated_nu_history'].append(comp_data['calculated_nu'].copy())
        
        # 每10000步保存一次数据
        if comp_data['step_count'] % 1000 == 0:
            print(f"[INFO] 准备保存数据 - 步数: {comp_data['step_count']}, 历史数据长度: {len(comp_data['isaaclab_eta_history'])}")
            save_comparison_data(env, 0)
    
    # 关键优化7：性能统计（可选）
    if not hasattr(env, '_batch_performance_counter'):
        env._batch_performance_counter = 0
    env._batch_performance_counter += 1
    
    # 关键优化8：直接设置平台位置和速度，确保所有平台完全同步
    # 使用 write_root_pose_to_sim 直接设置平台位置和姿态
    # 使用 write_root_velocity_to_sim 直接设置平台速度
    # 这样可以确保所有平台精确同步，避免误差累积和发散
    # 同时保持物理引擎的碰撞检测功能（平台仍然可以接触机器狗）
    
    # 计算世界坐标系速度
    # 重要：nu是在体坐标系下的速度，需要正确转换为世界坐标系
    # 使用shared_system.nu（体坐标系速度，已经更新为next_nu）
    next_nu_np = shared_system.nu.copy()
    # z方向速度正常使用，不进行强制限制
    
    next_nu_torch = torch.from_numpy(next_nu_np).to(
        dtype=current_quat.dtype,
        device=current_quat.device
    )
    next_nu_torch = next_nu_torch.unsqueeze(0).expand(num_envs, -1)  # [num_envs, 6]
    
    # 使用四元数转换体坐标系速度到世界坐标系
    # 线速度：从体坐标系到世界坐标系
    lin_vel_body = next_nu_torch[:, :3]  # [num_envs, 3] 体坐标系线速度
    lin_vel_world = math_utils.quat_apply(next_quat, lin_vel_body)  # [num_envs, 3] 世界坐标系线速度
    
    # 角速度：从体坐标系到世界坐标系（角速度也是向量，需要用四元数转换）
    ang_vel_body = next_nu_torch[:, 3:6]  # [num_envs, 3] 体坐标系角速度
    ang_vel_world = math_utils.quat_apply(next_quat, ang_vel_body)  # [num_envs, 3] 世界坐标系角速度
    
    # 组合线速度和角速度（都是世界坐标系）
    next_root_velocity = torch.cat([lin_vel_world, ang_vel_world], dim=-1)  # [num_envs, 6]
    
    # 直接设置平台位置和姿态（所有平台完全同步）
    asset.write_root_pose_to_sim(next_root_pose, env_ids=env_ids)
    # 同时设置速度（所有平台完全同步）
    asset.write_root_velocity_to_sim(next_root_velocity, env_ids=env_ids)

    # print("[INFO] 位置:", pose)
    # print("[INFO] 角速度:", asset.data.root_ang_vel_w[0,:])
    # # print("[INFO] 线速度:", asset.data.root_lin_vel_w[0,:])
    # # print("[INFO] 线加速度:", asset.data.body_lin_acc_w[0, :])
    # print("[INFO] 角加速度:", asset.data.body_ang_acc_w[0, :])
    
    # 重要：共享系统的内部状态已经在 step 函数内部更新了（self.eta 和 self.nu）
    # step 函数会：
    # 1. 用传入的 current_eta 和 current_nu 更新 self.eta 和 self.nu（第1541-1542行）
    # 2. 通过积分更新 self.eta 和 self.nu（第1578-1579行）
    # 3. 返回更新后的 self.eta 和 eta_dot
    # 所以这里不需要再次更新，下一帧会自动使用更新后的内部状态

    # ========================================================所有环境同一个平台代码===========================================
    # 注意：现在使用位置和速度设置，不再使用加速度设置
    # 所有平台的位置、速度、加速度完全同步，因为它们使用同一个VesselControlSystem实例


import numpy as np
def Rzyx(phi,theta,psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    """
    
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)
    
    R = np.array([
        [ cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth ],
        [ spsi*cth,  cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi ],
        [ -sth,      cth*sphi,                 cth*cphi ] ])

    return R
def Tzyx(phi,theta):
    """
    T = Tzyx(phi,theta) computes the Euler angle attitude
    transformation matrix T using the zyx convention
    """
    
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)    

    try: 
        T = np.array([
            [ 1,  sphi*sth/cth,  cphi*sth/cth ],
            [ 0,  cphi,          -sphi],
            [ 0,  sphi/cth,      cphi/cth] ])
        
    except ZeroDivisionError:  
        print ("Tzyx is singular for theta = +-90 degrees." )
        
    return T
    
def attitudeEuler(eta,nu,sampleTime):
    """
    eta = attitudeEuler(eta,nu,sampleTime) computes the generalized 
    position/Euler angles eta[k+1]
    """
   
    p_dot   = np.matmul( Rzyx(eta[3], eta[4], eta[5]), nu[0:3] )
    v_dot   = np.matmul( Tzyx(eta[3], eta[4]), nu[3:6] )

    # Forward Euler integration
    eta[0:3] = eta[0:3] + sampleTime * p_dot
    eta[3:6] = eta[3:6] + sampleTime * v_dot

    return eta





def randomize_rigid_body_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    """Randomize the scale of a rigid body asset in the USD stage.

    This function modifies the "xformOp:scale" property of all the prims corresponding to the asset.

    It takes a tuple or dictionary for the scale ranges. If it is a tuple, then the scaling along
    individual axis is performed equally. If it is a dictionary, the scaling is independent across each dimension.
    The keys of the dictionary are ``x``, ``y``, and ``z``. The values are tuples of the form ``(min, max)``.

    If the dictionary does not contain a key, the range is set to one for that axis.

    Relative child path can be used to randomize the scale of a specific child prim of the asset.
    For example, if the asset at prim path expression ``/World/envs/env_.*/Object`` has a child
    with the path ``/World/envs/env_.*/Object/mesh``, then the relative child path should be ``mesh`` or
    ``/mesh``.

    .. attention::
        Since this function modifies USD properties that are parsed by the physics engine once the simulation
        starts, the term should only be used before the simulation starts playing. This corresponds to the
        event mode named "usd". Using it at simulation time, may lead to unpredictable behaviors.

    .. note::
        When randomizing the scale of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """
    # check if sim is running
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if isinstance(asset, Articulation):
        raise ValueError(
            "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
            " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
            " each version of the articulation and using multi-asset spawning. For more details, refer to:"
            " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
        )

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # acquire stage
    stage = get_current_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    # sample scale values
    if isinstance(scale_range, dict):
        range_list = [scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    else:
        rand_samples = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu")
        rand_samples = rand_samples.repeat(1, 3)
    # convert to list for the for loop
    rand_samples = rand_samples.tolist()

    # apply the randomization to the parent if no relative child path is provided
    # this might be useful if user wants to randomize a particular mesh in the prim hierarchy
    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    # use sdf changeblock for faster processing of USD properties
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # path to prim to randomize
            prim_path = prim_paths[env_id] + relative_child_path
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # get the attribute to randomize
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            # if the scale attribute does not exist, create it
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            # set the new scale
            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            # ensure the operation is done in the right ordering if we created the scale attribute.
            # otherwise, we assume the scale attribute is already in the right order.
            # note: by default isaac sim follows this ordering for the transform stack so any asset
            #   created through it will have the correct ordering
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])


class randomize_rigid_body_material(ManagerTermBase):
    """Randomize the physics materials on all geometries of the asset.

    This function creates a set of physics materials with random static friction, dynamic friction, and restitution
    values. The number of materials is specified by ``num_buckets``. The materials are generated by sampling
    uniform random values from the given ranges.

    The material properties are then assigned to the geometries of the asset. The assignment is done by
    creating a random integer tensor of shape  (num_instances, max_num_shapes) where ``num_instances``
    is the number of assets spawned and ``max_num_shapes`` is the maximum number of shapes in the asset (over
    all bodies). The integer values are used as indices to select the material properties from the
    material buckets.

    If the flag ``make_consistent`` is set to ``True``, the dynamic friction is set to be less than or equal to
    the static friction. This obeys the physics constraint on friction values. However, it may not always be
    essential for the application. Thus, the flag is set to ``False`` by default.

    .. attention::
        This function uses CPU tensors to assign the material properties. It is recommended to use this function
        only during the initialization of the environment. Otherwise, it may lead to a significant performance
        overhead.

    .. note::
        PhysX only allows 64000 unique physics materials in the scene. If the number of materials exceeds this
        limit, the simulation will crash. Due to this reason, we sample the materials only once during initialization.
        Afterwards, these materials are randomly assigned to the geometries of the asset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # obtain number of shapes per body (needed for indexing the material properties correctly)
        # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
        #  per body. We use the physics simulation view to obtain the number of shapes per body.
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            # ensure the parsing is correct
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            # in this case, we don't need to do special indexing
            self.num_shapes_per_body = None

        # obtain parameters for sampling friction and restitution values
        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 1))

        # sample material properties from the given ranges
        # note: we only sample the materials once during initialization
        #   afterwards these are randomly assigned to the geometries of the asset
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")

        # ensure dynamic friction is always less than static friction
        make_consistent = cfg.params.get("make_consistent", False)
        if make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # randomly assign material IDs to the geometries
        total_num_shapes = self.asset.root_physx_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        # retrieve material buffer from the physics simulation
        materials = self.asset.root_physx_view.get_material_properties()

        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            # sample material properties from the given ranges
            for body_id in self.asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes x 3
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            # assign all the materials
            materials[env_ids] = material_samples[:]

        # apply to simulation
        self.asset.root_physx_view.set_material_properties(materials, env_ids)


class randomize_rigid_body_mass(ManagerTermBase):
    """Randomize the mass of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    If the ``recompute_inertia`` flag is set to ``True``, the function recomputes the inertia tensor of the bodies
    after setting the mass. This is useful when the mass is changed significantly, as the inertia tensor depends
    on the mass. It assumes the body is a uniform density object. If the body is not a uniform density object,
    the inertia tensor may not be accurate.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "mass_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["mass_distribution_params"], "mass_distribution_params", allow_zero=False
                )
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_rigid_body_mass' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # resolve body indices
        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        # get the current masses of the bodies (num_assets, num_bodies)
        masses = self.asset.root_physx_view.get_masses()

        # apply randomization on default values
        # this is to make sure when calling the function multiple times, the randomization is applied on the
        # default values and not the previously randomized values
        masses[env_ids[:, None], body_ids] = self.asset.data.default_mass[env_ids[:, None], body_ids].clone()

        # sample from the given range
        # note: we modify the masses in-place for all environments
        #   however, the setter takes care that only the masses of the specified environments are modified
        masses = _randomize_prop_by_op(
            masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
        )

        # set the mass into the physics simulation
        self.asset.root_physx_view.set_masses(masses, env_ids)

        # recompute inertia tensors if needed
        if recompute_inertia:
            # compute the ratios of the new masses to the initial masses
            ratios = masses[env_ids[:, None], body_ids] / self.asset.data.default_mass[env_ids[:, None], body_ids]
            # scale the inertia tensors by the the ratios
            # since mass randomization is done on default values, we can use the default inertia tensors
            inertias = self.asset.root_physx_view.get_inertias()
            if isinstance(self.asset, Articulation):
                # inertia has shape: (num_envs, num_bodies, 9) for articulation
                inertias[env_ids[:, None], body_ids] = (
                    self.asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
                )
            else:
                # inertia has shape: (num_envs, 9) for rigid object
                inertias[env_ids] = self.asset.data.default_inertia[env_ids] * ratios
            # set the inertia tensors into the physics simulation
            self.asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[env_ids[:, None], body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_rigid_body_collider_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    rest_offset_distribution_params: tuple[float, float] | None = None,
    contact_offset_distribution_params: tuple[float, float] | None = None,
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the collider parameters of rigid bodies in an asset by adding, scaling, or setting random values.

    This function allows randomizing the collider parameters of the asset, such as rest and contact offsets.
    These correspond to the physics engine collider properties that affect the collision checking.

    The function samples random values from the given distribution parameters and applies the operation to
    the collider properties. It then sets the values into the physics simulation. If the distribution parameters
    are not provided for a particular property, the function does not modify the property.

    Currently, the distribution parameters are applied as absolute values.

    .. tip::
        This function uses CPU tensors to assign the collision properties. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")

    # sample collider properties from the given ranges and set into the physics simulation
    # -- rest offsets
    if rest_offset_distribution_params is not None:
        rest_offset = asset.root_physx_view.get_rest_offsets().clone()
        rest_offset = _randomize_prop_by_op(
            rest_offset,
            rest_offset_distribution_params,
            None,
            slice(None),
            operation="abs",
            distribution=distribution,
        )
        asset.root_physx_view.set_rest_offsets(rest_offset, env_ids.cpu())
    # -- contact offsets
    if contact_offset_distribution_params is not None:
        contact_offset = asset.root_physx_view.get_contact_offsets().clone()
        contact_offset = _randomize_prop_by_op(
            contact_offset,
            contact_offset_distribution_params,
            None,
            slice(None),
            operation="abs",
            distribution=distribution,
        )
        asset.root_physx_view.set_contact_offsets(contact_offset, env_ids.cpu())


def randomize_physics_scene_gravity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    gravity_distribution_params: tuple[list[float], list[float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize gravity by adding, scaling, or setting random values.

    This function allows randomizing gravity of the physics scene. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the
    operation.

    The distribution parameters are lists of two elements each, representing the lower and upper bounds of the
    distribution for the x, y, and z components of the gravity vector. The function samples random values for each
    component independently.

    .. attention::
        This function applied the same gravity for all the environments.

    .. tip::
        This function uses CPU tensors to assign gravity.
    """
    # get the current gravity
    gravity = torch.tensor(env.sim.cfg.gravity, device="cpu").unsqueeze(0)
    dist_param_0 = torch.tensor(gravity_distribution_params[0], device="cpu")
    dist_param_1 = torch.tensor(gravity_distribution_params[1], device="cpu")
    gravity = _randomize_prop_by_op(
        gravity,
        (dist_param_0, dist_param_1),
        None,
        slice(None),
        operation=operation,
        distribution=distribution,
    )
    # unbatch the gravity tensor into a list
    gravity = gravity[0].tolist()

    # set the gravity into the physics simulation
    physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view
    physics_sim_view.set_gravity(carb.Float3(*gravity))


class randomize_actuator_gains(ManagerTermBase):
    """Randomize the actuator gains in an articulation by adding, scaling, or setting random values.

    This function allows randomizing the actuator stiffness and damping gains.

    The function samples random values from the given distribution parameters and applies the operation to the joint properties.
    It then sets the values into the actuator models. If the distribution parameters are not provided for a particular property,
    the function does not modify the property.

    .. tip::
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        In such cases, it is recommended to use this function only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["damping_distribution_params"], "damping_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_actuator_gains' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # Resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        def randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
            return _randomize_prop_by_op(
                data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
            )

        # Loop through actuators and randomize gains
        for actuator in self.asset.actuators.values():
            if isinstance(self.asset_cfg.joint_ids, slice):
                # we take all the joints of the actuator
                actuator_indices = slice(None)
                if isinstance(actuator.joint_indices, slice):
                    global_indices = slice(None)
                elif isinstance(actuator.joint_indices, torch.Tensor):
                    global_indices = actuator.joint_indices.to(self.asset.device)
                else:
                    raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
            elif isinstance(actuator.joint_indices, slice):
                # we take the joints defined in the asset config
                global_indices = actuator_indices = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
            else:
                # we take the intersection of the actuator joints and the asset config joints
                actuator_joint_indices = actuator.joint_indices
                asset_joint_ids = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
                # the indices of the joints in the actuator that have to be randomized
                actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
                if len(actuator_indices) == 0:
                    continue
                # maps actuator indices that have to be randomized to global joint indices
                global_indices = actuator_joint_indices[actuator_indices]
            # Randomize stiffness
            if stiffness_distribution_params is not None:
                stiffness = actuator.stiffness[env_ids].clone()
                stiffness[:, actuator_indices] = self.asset.data.default_joint_stiffness[env_ids][
                    :, global_indices
                ].clone()
                randomize(stiffness, stiffness_distribution_params)
                actuator.stiffness[env_ids] = stiffness
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_stiffness_to_sim(
                        stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids
                    )
            # Randomize damping
            if damping_distribution_params is not None:
                damping = actuator.damping[env_ids].clone()
                damping[:, actuator_indices] = self.asset.data.default_joint_damping[env_ids][:, global_indices].clone()
                randomize(damping, damping_distribution_params)
                actuator.damping[env_ids] = damping
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


class randomize_joint_parameters(ManagerTermBase):
    """Randomize the simulated joint parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the joint parameters of the asset. These correspond to the physics engine
    joint properties that affect the joint behavior. The properties include the joint friction coefficient, armature,
    and joint position limits.

    The function samples random values from the given distribution parameters and applies the operation to the
    joint properties. It then sets the values into the physics simulation. If the distribution parameters are
    not provided for a particular property, the function does not modify the property.

    .. tip::
        This function uses CPU tensors to assign the joint properties. It is recommended to use this function
        only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "friction_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["friction_distribution_params"], "friction_distribution_params")
            if "armature_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["armature_distribution_params"], "armature_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        # resolve joint indices
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)  # for optimization purposes
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        # sample joint properties from the given ranges and set into the physics simulation
        # joint friction coefficient
        if friction_distribution_params is not None:
            friction_coeff = _randomize_prop_by_op(
                self.asset.data.default_joint_friction_coeff.clone(),
                friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_friction_coefficient_to_sim(
                friction_coeff[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )

        # joint armature
        if armature_distribution_params is not None:
            armature = _randomize_prop_by_op(
                self.asset.data.default_joint_armature.clone(),
                armature_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_armature_to_sim(
                armature[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )

        # joint position limits
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            joint_pos_limits = self.asset.data.default_joint_pos_limits.clone()
            # -- randomize the lower limits
            if lower_limit_distribution_params is not None:
                joint_pos_limits[..., 0] = _randomize_prop_by_op(
                    joint_pos_limits[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            # -- randomize the upper limits
            if upper_limit_distribution_params is not None:
                joint_pos_limits[..., 1] = _randomize_prop_by_op(
                    joint_pos_limits[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # extract the position limits for the concerned joints
            joint_pos_limits = joint_pos_limits[env_ids[:, None], joint_ids]
            if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater"
                    " than upper joint limits. Please check the distribution parameters for the joint position limits."
                )
            # set the position limits into the physics simulation
            self.asset.write_joint_position_limit_to_sim(
                joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
            )


class randomize_fixed_tendon_parameters(ManagerTermBase):
    """Randomize the simulated fixed tendon parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the fixed tendon parameters of the asset.
    These correspond to the physics engine tendon properties that affect the joint behavior.

    The function samples random values from the given distribution parameters and applies the operation to the tendon properties.
    It then sets the values into the physics simulation. If the distribution parameters are not provided for a
    particular property, the function does not modify the property.

    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["damping_distribution_params"], "damping_distribution_params")
            if "limit_stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["limit_stiffness_distribution_params"], "limit_stiffness_distribution_params"
                )
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        limit_stiffness_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        rest_length_distribution_params: tuple[float, float] | None = None,
        offset_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        # resolve joint indices
        if self.asset_cfg.fixed_tendon_ids == slice(None):
            tendon_ids = slice(None)  # for optimization purposes
        else:
            tendon_ids = torch.tensor(self.asset_cfg.fixed_tendon_ids, dtype=torch.int, device=self.asset.device)

        # sample tendon properties from the given ranges and set into the physics simulation
        # stiffness
        if stiffness_distribution_params is not None:
            stiffness = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_stiffness.clone(),
                stiffness_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_stiffness(stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

        # damping
        if damping_distribution_params is not None:
            damping = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_damping.clone(),
                damping_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_damping(damping[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

        # limit stiffness
        if limit_stiffness_distribution_params is not None:
            limit_stiffness = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_limit_stiffness.clone(),
                limit_stiffness_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_limit_stiffness(
                limit_stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids
            )

        # position limits
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            limit = self.asset.data.default_fixed_tendon_pos_limits.clone()
            # -- lower limit
            if lower_limit_distribution_params is not None:
                limit[..., 0] = _randomize_prop_by_op(
                    limit[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    tendon_ids,
                    operation=operation,
                    distribution=distribution,
                )
            # -- upper limit
            if upper_limit_distribution_params is not None:
                limit[..., 1] = _randomize_prop_by_op(
                    limit[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    tendon_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # check if the limits are valid
            tendon_limits = limit[env_ids[:, None], tendon_ids]
            if (tendon_limits[..., 0] > tendon_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_fixed_tendon_parameters' is setting lower tendon limits that are"
                    " greater than upper tendon limits."
                )
            self.asset.set_fixed_tendon_position_limit(tendon_limits, tendon_ids, env_ids)

        # rest length
        if rest_length_distribution_params is not None:
            rest_length = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_rest_length.clone(),
                rest_length_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_rest_length(rest_length[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

        # offset
        if offset_distribution_params is not None:
            offset = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_offset.clone(),
                offset_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_offset(offset[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

        # write the fixed tendon properties into the simulation
        self.asset.write_fixed_tendon_properties_to_sim(tendon_ids, env_ids)


def apply_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # GMY platform
    if "platform" in env.scene.keys():
        # # platform 
        platform = env.scene["platform"]

        # # 固定在平台上方某高度，比如 2 * platform_z + 0.6
        platform_pos = platform.data.root_com_pos_w[env_ids]
        height_offset = 0.5 * platform.cfg.spawn.size[2] + 0.6  # 米

        positions = platform_pos.clone()
        positions[:, 2] += height_offset
        positions[:, 0:2] += rand_samples[:, 0:2] + 10
    else:
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_with_random_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root position and velocities sampled randomly within the given ranges
    and the asset root orientation sampled randomly from the SO(3).

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation uniformly from the SO(3) and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of position ranges for each axis. The keys of the dictionary are ``x``,
      ``y``, and ``z``. The orientation is sampled uniformly from the SO(3).
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples
    orientations = math_utils.random_orientation(len(env_ids), device=asset.device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_from_terrain(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state by sampling a random valid pose from the terrain.

    This function samples a random valid pose(based on flat patches) from the terrain and sets the root state
    of the asset to this position. The function also samples random velocities from the given ranges and sets them
    into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of pose ranges for each axis. The keys of the dictionary are ``roll``,
      ``pitch``, and ``yaw``. The position is sampled from the flat patches of the terrain.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.

    Note:
        The function expects the terrain to have valid flat patches under the key "init_pos". The flat patches
        are used to sample the random pose for the robot.

    Raises:
        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
    """
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # obtain all flat patches corresponding to the valid poses
    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample random valid poses
    ids = torch.randint(0, valid_positions.shape[2], size=(len(env_ids),), device=env.device)
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_state[env_ids, :3]

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = asset.data.default_root_state[env_ids, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_nodal_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset nodal state to a random position and velocity uniformly within the given ranges.

    This function randomizes the nodal position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default nodal position, before setting
      them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis. The keys of the
    dictionary are ``x``, ``y``, ``z``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: DeformableObject = env.scene[asset_cfg.name]
    # get default root state
    nodal_state = asset.data.default_nodal_state_w[env_ids].clone()

    # position
    range_list = [position_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 1, 3), device=asset.device)

    nodal_state[..., :3] += rand_samples

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 1, 3), device=asset.device)

    nodal_state[..., 3:] += rand_samples

    # set into the physics simulation
    asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor, reset_joint_targets: bool = False):
    """Reset the scene to the default state specified in the scene configuration.

    If :attr:`reset_joint_targets` is True, the joint position and velocity targets of the articulations are
    also reset to their default values. This might be useful for some cases to clear out any previously set targets.
    However, this is not the default behavior as based on our experience, it is not always desired to reset
    targets to default values, especially when the targets should be handled by action terms and not event terms.
    """
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        # reset joint targets if required
        if reset_joint_targets:
            articulation_asset.set_joint_position_target(default_joint_pos, env_ids=env_ids)
            articulation_asset.set_joint_velocity_target(default_joint_vel, env_ids=env_ids)
    # deformable objects
    for deformable_object in env.scene.deformable_objects.values():
        # obtain default and set into the physics simulation
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


class randomize_visual_texture_material(ManagerTermBase):
    """Randomize the visual texture of bodies on an asset using Replicator API.

    This function randomizes the visual texture of the bodies of the asset using the Replicator API.
    The function samples random textures from the given texture paths and applies them to the bodies
    of the asset. The textures are projected onto the bodies and rotated by the given angles.

    .. note::
        The function assumes that the asset follows the prim naming convention as:
        "{asset_prim_path}/{body_name}/visuals" where the body name is the name of the body to
        which the texture is applied. This is the default prim ordering when importing assets
        from the asset converters in Isaac Lab.

    .. note::
        When randomizing the texture of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual texture material with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")

        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")

        # obtain the asset entity
        asset = env.scene[asset_cfg.name]

        # join all bodies in the asset
        body_names = asset_cfg.body_names
        if isinstance(body_names, str):
            body_names_regex = body_names
        elif isinstance(body_names, list):
            body_names_regex = "|".join(body_names)
        else:
            body_names_regex = ".*"

        # create the affected prim path
        # Check if the pattern with '/visuals' yields results when matching `body_names_regex`.
        # If not, fall back to a broader pattern without '/visuals'.
        asset_main_prim_path = asset.cfg.prim_path
        pattern_with_visuals = f"{asset_main_prim_path}/{body_names_regex}/visuals"
        # Use sim_utils to check if any prims currently match this pattern
        matching_prims = sim_utils.find_matching_prim_paths(pattern_with_visuals)
        if matching_prims:
            # If matches are found, use the pattern with /visuals
            prim_path = pattern_with_visuals
        else:
            # If no matches found, fall back to the broader pattern without /visuals
            # This pattern (e.g., /World/envs/env_.*/Table/.*) should match visual prims
            # whether they end in /visuals or have other structures.
            prim_path = f"{asset_main_prim_path}/.*"
            carb.log_info(
                f"Pattern '{pattern_with_visuals}' found no prims. Falling back to '{prim_path}' for texture"
                " randomization."
            )

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            texture_paths = cfg.params.get("texture_paths")
            event_name = cfg.params.get("event_name")
            texture_rotation = cfg.params.get("texture_rotation", (0.0, 0.0))

            # convert from radians to degrees
            texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

            # Create the omni-graph node for the randomization term
            def rep_texture_randomization():
                prims_group = rep.get.prims(path_pattern=prim_path)

                with prims_group:
                    rep.randomizer.texture(
                        textures=texture_paths,
                        project_uvw=True,
                        texture_rotate=rep.distribution.uniform(*texture_rotation),
                    )
                return prims_group.node

            # Register the event to the replicator
            with rep.trigger.on_custom_event(event_name=event_name):
                rep_texture_randomization()
        else:
            # acquire stage
            stage = get_current_stage()
            prims_group = rep.functional.get.prims(path_pattern=prim_path, stage=stage)

            num_prims = len(prims_group)
            # rng that randomizes the texture and rotation
            self.texture_rng = rep.rng.ReplicatorRNG()

            # Create the material first and bind it to the prims
            for i, prim in enumerate(prims_group):
                # Disable instancble
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)

            # TODO: Should we specify the value when creating the material?
            self.material_prims = rep.functional.create_batch.material(
                mdl="OmniPBR.mdl", bind_prims=prims_group, count=num_prims, project_uvw=True
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
    ):
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            rep.utils.send_og_event(event_name)
        else:
            # read parameters from the configuration
            texture_paths = texture_paths if texture_paths else self._cfg.params.get("texture_paths")
            texture_rotation = (
                texture_rotation if texture_rotation else self._cfg.params.get("texture_rotation", (0.0, 0.0))
            )

            # convert from radians to degrees
            texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

            num_prims = len(self.material_prims)
            random_textures = self.texture_rng.generator.choice(texture_paths, size=num_prims)
            random_rotations = self.texture_rng.generator.uniform(
                texture_rotation[0], texture_rotation[1], size=num_prims
            )

            # modify the material properties
            rep.functional.modify.attribute(self.material_prims, "diffuse_texture", random_textures)
            rep.functional.modify.attribute(self.material_prims, "texture_rotate", random_rotations)


class randomize_visual_color(ManagerTermBase):
    """Randomize the visual color of bodies on an asset using Replicator API.

    This function randomizes the visual color of the bodies of the asset using the Replicator API.
    The function samples random colors from the given colors and applies them to the bodies
    of the asset.

    The function assumes that the asset follows the prim naming convention as:
    "{asset_prim_path}/{mesh_name}" where the mesh name is the name of the mesh to
    which the color is applied. For instance, if the asset has a prim path "/World/asset"
    and a mesh named "body_0/mesh", the prim path for the mesh would be
    "/World/asset/body_0/mesh".

    The colors can be specified as a list of tuples of the form ``(r, g, b)`` or as a dictionary
    with the keys ``r``, ``g``, ``b`` and values as tuples of the form ``(low, high)``.
    If a dictionary is used, the function will sample random colors from the given ranges.

    .. note::
        When randomizing the color of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the randomization term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        mesh_name: str = cfg.params.get("mesh_name", "")  # type: ignore

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual color with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # obtain the asset entity
        asset = env.scene[asset_cfg.name]

        # create the affected prim path
        if not mesh_name.startswith("/"):
            mesh_name = "/" + mesh_name
        mesh_prim_path = f"{asset.cfg.prim_path}{mesh_name}"
        # TODO: Need to make it work for multiple meshes.

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            colors = cfg.params.get("colors")
            event_name = cfg.params.get("event_name")

            # parse the colors into replicator format
            if isinstance(colors, dict):
                # (r, g, b) - low, high --> (low_r, low_g, low_b) and (high_r, high_g, high_b)
                color_low = [colors[key][0] for key in ["r", "g", "b"]]
                color_high = [colors[key][1] for key in ["r", "g", "b"]]
                colors = rep.distribution.uniform(color_low, color_high)
            else:
                colors = list(colors)

            # Create the omni-graph node for the randomization term
            def rep_color_randomization():
                prims_group = rep.get.prims(path_pattern=mesh_prim_path)
                with prims_group:
                    rep.randomizer.color(colors=colors)

                return prims_group.node

            # Register the event to the replicator
            with rep.trigger.on_custom_event(event_name=event_name):
                rep_color_randomization()
        else:
            stage = get_current_stage()
            prims_group = rep.functional.get.prims(path_pattern=mesh_prim_path, stage=stage)

            num_prims = len(prims_group)
            self.color_rng = rep.rng.ReplicatorRNG()

            # Create the material first and bind it to the prims
            for i, prim in enumerate(prims_group):
                # Disable instancble
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)

            # TODO: Should we specify the value when creating the material?
            self.material_prims = rep.functional.create_batch.material(
                mdl="OmniPBR.mdl", bind_prims=prims_group, count=num_prims, project_uvw=True
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
        mesh_name: str = "",
    ):
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.

        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            rep.utils.send_og_event(event_name)
        else:
            colors = colors if colors else self._cfg.params.get("colors")

            # parse the colors into replicator format
            if isinstance(colors, dict):
                # (r, g, b) - low, high --> (low_r, low_g, low_b) and (high_r, high_g, high_b)
                color_low = [colors[key][0] for key in ["r", "g", "b"]]
                color_high = [colors[key][1] for key in ["r", "g", "b"]]
                colors = [color_low, color_high]
            else:
                colors = list(colors)

            num_prims = len(self.material_prims)
            random_colors = self.color_rng.generator.uniform(colors[0], colors[1], size=(num_prims, 3))

            rep.functional.modify.attribute(self.material_prims, "diffuse_color_constant", random_colors)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def save_comparison_data(env, env_id):
    """保存对比数据到文件"""
    import numpy as np
    import os
    
    if not hasattr(env, '_comparison_data') or env_id not in env._comparison_data:
        return
    
    comp_data = env._comparison_data[env_id]
    
    # 创建保存目录
    save_dir = "/home/user/IsaacLab/comparison_data"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存数据
    filename = f"{save_dir}/env_{env_id}_step_{comp_data['step_count']}.npz"
    np.savez(filename,
             isaaclab_eta_history=np.array(comp_data['isaaclab_eta_history']),
             isaaclab_nu_history=np.array(comp_data['isaaclab_nu_history']),
             calculated_eta_history=np.array(comp_data['calculated_eta_history']),
             calculated_nu_history=np.array(comp_data['calculated_nu_history']),
             step_count=comp_data['step_count'])
    
    print(f"对比数据已保存到: {filename}")


def _validate_scale_range(
    params: tuple[float, float] | None,
    name: str,
    *,
    allow_negative: bool = False,
    allow_zero: bool = True,
) -> None:
    """
    Validates a (low, high) tuple used in scale-based randomization.

    This function ensures the tuple follows expected rules when applying a 'scale'
    operation. It performs type and value checks, optionally allowing negative or
    zero lower bounds.

    Args:
        params (tuple[float, float] | None): The (low, high) range to validate. If None,
            validation is skipped.
        name (str): The name of the parameter being validated, used for error messages.
        allow_negative (bool, optional): If True, allows the lower bound to be negative.
            Defaults to False.
        allow_zero (bool, optional): If True, allows the lower bound to be zero.
            Defaults to True.

    Raises:
        TypeError: If `params` is not a tuple of two numbers.
        ValueError: If the lower bound is negative or zero when not allowed.
        ValueError: If the upper bound is less than the lower bound.

    Example:
        _validate_scale_range((0.5, 1.5), "mass_scale")
    """
    if params is None:  # caller didn’t request randomisation for this field
        return
    low, high = params
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise TypeError(f"{name}: expected (low, high) to be a tuple of numbers, got {params}.")
    if not allow_negative and not allow_zero and low <= 0:
        raise ValueError(f"{name}: lower bound must be > 0 when using the 'scale' operation (got {low}).")
    if not allow_negative and allow_zero and low < 0:
        raise ValueError(f"{name}: lower bound must be ≥ 0 when using the 'scale' operation (got {low}).")
    if high < low:
        raise ValueError(f"{name}: upper bound ({high}) must be ≥ lower bound ({low}).")

