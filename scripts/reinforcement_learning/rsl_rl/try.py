# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import pandas as pd
import numpy as np


import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 解析命令行参数
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    # 是否录制视频
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    # 视频录制时长（步数）
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    # 是否禁用 Fabric，改用 USD I/O 操作
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
    # 模拟的环境数量
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    # 运行的任务名称
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    # 是否使用预训练的检查点
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
    # 是否以实时模式运行
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# 添加 RSL-RL 相关参数（如 PPO 训练参数）
cli_args.add_rsl_rl_args(parser)
# 添加 AppLauncher 相关参数（如 Isaac Sim 运行参数）
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli = parser.parse_args()
#  启用摄像头 进行录制
if args_cli.video:
    args_cli.enable_cameras = True

# 创建 AppLauncher 实例，启动 Omniverse（Isaac Sim 物理模拟器）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# 强化学习环境接口
import gymnasium as gym
import os
import time
import torch

# RSL-RL 的强化学习 PPO 训练器
from rsl_rl.runners import OnPolicyRunner
# 多智能体强化学习（MARL）
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
# 导出模型为 JIT 或 ONNX 格式
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """加载 RSL-RL 训练好的智能体，在 Isaac Sim 进行推理"""
    # 解析任务（命令行参数）
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # 解析强化学习训练的超参数（如 PPO 配置）
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 选择检查点，在整个代码根目录下的，即IsaacLab/logs/rsl_rl/下
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # 使用 官方预训练模型
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    # 使用 本地训练好的模型
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    # 读取 最新的实验检查点（默认）
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # 创建 Gym 环境（可视化模式 rgb_array 用于录制视频）
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 转换为单智能体环境（如 RSL-RL 不支持多智能体）
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    # 创建环境
        # 如果开启视频录制，则包装环境，保存视频
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
        # 包装环境以适配 RSL-RL 算法
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # 加载训练好的 PPO 代理
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        # 这行代码从指定路径 resume_path 加载之前训练好的PPO模型
    ppo_runner.load(resume_path)

        # 从PPO代理中提取出用于推理的策略
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # 导出策略为ONNX/JIT格式
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        # 定义了导出模型的存储路径，存储路径是在训练检查点的文件夹中创建一个 exported 目录
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
        # 将PPO代理中的策略（actor_critic）以及观察归一化器（obs_normalizer）导出为JIT（Just-In-Time）格式，
        # 并保存为 policy.pt 文件
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )
    # 获取模拟环境的物理时间步长
    dt = env.unwrapped.physics_dt

    # 循环执行策略推理，控制智能体在 Isaac Sim 运行
    obs, _ = env.get_observations()
    # 计时器
    timestep = 0


    # GMY 新增：初始化数据记录变量
    import isaaclab.utils.math as math_utils
    from scipy.spatial.transform import Rotation as Rotate 

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():                    # 记录当前现实，用于计算每一步的运行时间，确保实时仿真    
            actions = policy(obs)                       # 根据观测值 obs 计算动作   
            obs, _, _, _ = env.step(actions)            # 计算环境的下一步状态
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:       # 当达到设定的 video_length 时，退出循环，停止录制
                break
        sleep_time = dt - (time.time() - start_time)    # 计算本次推理的运行时间并与仿真时间步长 dt 比较

        # GMY =========================================================================================================
        # 获取机器人实例
        robot = env.unwrapped.scene["robot"]  # Articulation 对象
        # 获取platform刚体信息
        platform = env.unwrapped.scene["platform"]  # RigidObject 对象

        # 计算各个变量================================================================================================
        def calculate_value(input_values = None, output_variable_name="a_rn_r"):
            if input_values is None:
                y = 1
            else:
                x = input_values[0]
                y = input_values[1]

            # ---- Step 1: 准备四元数与基本变量 ----
            R_r_n, R_p_n = robot.data.root_quat_w[0], platform.data.root_quat_w[0]
            R_p_r = math_utils.quat_mul(math_utils.quat_conjugate(R_r_n), R_p_n)
            
            a_pn_p = math_utils.quat_rotate_inverse(R_p_n, platform.data.body_lin_acc_w[0, 0])

            w_dot_rn_r = math_utils.quat_rotate_inverse(R_r_n, robot.data.body_ang_acc_w[0, 0])
            w_rn_r = math_utils.quat_rotate_inverse(R_r_n, robot.data.root_ang_vel_w[0])

            p_ir_r = math_utils.quat_rotate_inverse(
                R_r_n,
                (robot.data.body_pos_w[0, y] - robot.data.root_pos_w[0])
            )

            v_ir_n = (robot.data.body_lin_vel_w[0, y] - robot.data.root_lin_vel_w[0])
            p_dot_ir_r = math_utils.quat_rotate_inverse(R_r_n, v_ir_n) - torch.cross(w_rn_r, p_ir_r)

            a_ir_n = (robot.data.body_lin_acc_w[0, y] - robot.data.body_lin_acc_w[0, 0])
            p_ddot_ir_r = (
                math_utils.quat_rotate_inverse(R_r_n, a_ir_n)
                - torch.cross(w_dot_rn_r, p_ir_r)
                - torch.cross(w_rn_r, torch.cross(w_rn_r, p_ir_r))
                - 2 * torch.cross(w_rn_r, p_dot_ir_r)
            )

            w_dot_pn_p = math_utils.quat_rotate_inverse(R_p_n, platform.data.body_ang_acc_w[0, 0])
            w_pn_p = math_utils.quat_rotate_inverse(R_p_n, platform.data.root_ang_vel_w[0])

            p_ip_p = math_utils.quat_rotate_inverse(
                R_p_n,
                (robot.data.body_pos_w[0, y] - platform.data.root_pos_w[0])
            )

            v_ip_n = (robot.data.body_lin_vel_w[0, y] - platform.data.root_lin_vel_w[0])
            p_dot_ip_p = math_utils.quat_rotate_inverse(R_p_n, v_ip_n) - torch.cross(w_pn_p, p_ip_p)

            a_ip_n = (robot.data.body_lin_acc_w[0, y] - platform.data.body_lin_acc_w[0, 0])
            p_ddot_ip_p = (
                math_utils.quat_rotate_inverse(R_p_n, a_ip_n)
                - torch.cross(w_dot_pn_p, p_ip_p)
                - torch.cross(w_pn_p, torch.cross(w_pn_p, p_ip_p))
                - 2 * torch.cross(w_pn_p, p_dot_ip_p)
            )


            # ---- Step 2: 计算最终表达式 ----
            a_rn_r = (
                math_utils.quat_rotate(R_p_r, a_pn_p)
                - torch.cross(w_dot_rn_r, p_ir_r)
                - torch.cross(w_rn_r, torch.cross(w_rn_r, p_ir_r))
                - 2 * torch.cross(w_rn_r, p_dot_ir_r)
                - p_ddot_ir_r
                + math_utils.quat_rotate(R_p_r, torch.cross(w_dot_pn_p, p_ip_p) + 
                                         torch.cross(w_pn_p, torch.cross(w_pn_p, p_ip_p)))
                + math_utils.quat_rotate(R_p_r, p_ddot_ip_p)
                + 2 * math_utils.quat_rotate(R_p_r, torch.cross(w_pn_p, p_dot_ip_p))
            )

            # 这是a_rn_r_true
            a_rn_r_true = math_utils.quat_rotate_inverse(robot.data.root_quat_w[0], robot.data.body_lin_acc_w[0, 0])
            if y <= 12:
                joint_pva = torch.tensor(
                    [robot.data.joint_pos[0][y].item(), 
                    robot.data.joint_vel[0][y].item(), 
                    robot.data.joint_acc[0][y].item()],
                    device=torch.device("cuda:0"),
                    dtype=torch.float32
                    )
            
            p_pr_r = math_utils.quat_rotate_inverse(R_r_n, platform.data.root_pos_w[0] - robot.data.root_pos_w[0])
            w_pr_r = math_utils.quat_rotate_inverse(R_r_n, platform.data.root_ang_vel_w[0] - robot.data.root_ang_vel_w[0])

            R_i_r = math_utils.quat_mul(math_utils.quat_conjugate(R_r_n), robot.data.body_quat_w[0, y])
             
            # ---- Step 3: 返回指定变量 ----
            local_vars = locals()
            if output_variable_name in local_vars:
                return local_vars[output_variable_name]
            else:
                raise ValueError(f"[ERROR] Variable '{output_variable_name}' not found in local scope.")
            
        # 计算各个变量================================================================================================

        # # 保存数据===================================================================================================
        save_dir = "save_data/save_data5/"
        os.makedirs(save_dir, exist_ok=True)
        
        # # 定义保存函数
        def save_tensor(tensor, save_dir, filename, columns=None):
            # 确保目录存在
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  # 创建目录，如果不存在
            
            # 创建完整文件路径
            filepath = os.path.join(save_dir, filename)
            
            # 将Tensor转换为numpy数组并保存为一行
            array = tensor.detach().cpu().numpy().reshape(1, -1) 
            df_new = pd.DataFrame(array, columns=columns)
            
            # 判断文件是否存在
            if os.path.exists(filepath):
                # 如果存在，追加数据
                try:
                    df_existing = pd.read_excel(filepath, engine="openpyxl")
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                except Exception as e:
                    print(f"Error reading the existing file: {e}. Overwriting the file.")
                    df_combined = df_new
            else:
                # 如果不存在，创建新文件
                df_combined = df_new
            
            # 保存数据到Excel文件
            try:
                with pd.ExcelWriter(filepath, engine="openpyxl", mode="w") as writer:
                    df_combined.to_excel(writer, index=False)
                # print(f"Data saved successfully to {filepath}")
            except Exception as e:
                print(f"Error saving to {filepath}: {e}")
        def Euler_angeles(R_p_n):
            quat_xyzw = [R_p_n[1].item(), R_p_n[2].item(), R_p_n[3].item(), R_p_n[0].item()]
            r = Rotate.from_quat(quat_xyzw)
            euler_deg = r.as_euler("xyz", degrees=True)
            return torch.tensor(euler_deg, dtype=R_p_n.dtype, device=R_p_n.device)

        # # # # # 保存每个数组 
        save_tensor(calculate_value([0, 0], 'R_p_n'), save_dir, "R_p_n.xlsx")
        save_tensor(calculate_value([0, 0], 'R_r_n'), save_dir, "R_r_n.xlsx")
        save_tensor(calculate_value([0, 0], 'R_p_r'), save_dir, "R_p_r.xlsx")

        # save_tensor(Euler_angeles(calculate_value([0, 0], 'R_p_n')), save_dir, "Theta_p_n.xlsx")
        # save_tensor(Euler_angeles(calculate_value([0, 0], 'R_r_n')), save_dir, "Theta_r_n.xlsx")
        # save_tensor(Euler_angeles(calculate_value([0, 0], 'R_p_r')), save_dir, "Theta_p_r.xlsx")

        save_tensor(calculate_value([0, 0], 'w_rn_r'), save_dir, "w_rn_r.xlsx")
        save_tensor(calculate_value([0, 0], 'w_pn_p'), save_dir, "w_pn_p.xlsx")
        save_tensor(calculate_value([0, 0], 'w_pr_r'), save_dir, "w_pr_r.xlsx")

        save_tensor(calculate_value([0, 0], 'w_dot_rn_r'), save_dir, "w_dot_rn_r.xlsx")
        save_tensor(calculate_value([0, 0], 'w_dot_pn_p'), save_dir, "w_dot_pn_p.xlsx")

        save_tensor(calculate_value([0, 0], 'a_pn_p'), save_dir, "a_pn_p.xlsx")
        save_tensor(calculate_value([0, 0], 'a_rn_r'), save_dir, "a_rn_r.xlsx")

        for k in range(15, 16):
            save_tensor(calculate_value([0, k], 'p_ir_r'), save_dir, f"p_ir_r{k}.xlsx")
            save_tensor(calculate_value([0, k], 'p_dot_ir_r'), save_dir, f"v_ir_r{k}.xlsx")
            save_tensor(calculate_value([0, k], 'p_ddot_ir_r'), save_dir, f"a_ir_r{k}.xlsx")

            save_tensor(calculate_value([0, k], 'p_ip_p'), save_dir, f"p_ip_p{k}.xlsx")
            save_tensor(calculate_value([0, k], 'p_dot_ip_p'), save_dir, f"v_ip_p{k}.xlsx")
            save_tensor(calculate_value([0, k], 'p_ddot_ip_p'), save_dir, f"a_ip_p{k}.xlsx")
        # for k in range(12):
        #     save_tensor(calculate_value([0, k], 'joint_pva'), save_dir, f"joint_pva{k}.xlsx")

        save_tensor(calculate_value([0, 0], 'p_pr_r'), save_dir, "p_pr_r.xlsx")
        



        # # 保存数据===================================================================================================

        # # # 计算相对角速度 w_pr_r=====================================================================================
        # def get_cross_matrix(v):
        #     # v: torch tensor with shape [3]
        #     return torch.tensor([
        #         [0, -v[2], v[1]],
        #         [v[2], 0, -v[0]],
        #         [-v[1], v[0], 0]
        #     ], dtype=torch.float32)

        # def compute_relative_angular_velocity():
        #     R_r_n = robot.data.root_quat_w[0]

        #     A = []
        #     b = []

        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确保所有张量都在相同设备上

        #     # 迭代三条腿，计算腿之间的位置和速度差
        #     for i in range(3):  # 只考虑腿1、腿2和腿2、腿3
        #         # 获取两条腿的位置和速度
        #         p_1r_r = calculate_value([0, 15 + i], 'p_ir_r')
        #         p_2r_r = calculate_value([0, 15 + i + 1], 'p_ir_r')
        #         v_1r_r = calculate_value([0, 15 + i], 'p_dot_ir_r') - math_utils.quat_rotate(
        #             calculate_value([0, 0], 'R_p_r'),
        #             calculate_value([0, 15 + i], 'p_dot_ip_p')
        #         )
        #         v_2r_r = calculate_value([0, 15 + i + 1], 'p_dot_ir_r') - math_utils.quat_rotate(
        #             calculate_value([0, 0], 'R_p_r'),
        #             calculate_value([0, 15 + i + 1], 'p_dot_ip_p')
        #         )   
            
        #         p_root_p_w = platform.data.root_pos_w[0].to(device)
        #         v_root_p_w = platform.data.root_lin_vel_w[0].to(device)
        #         # 计算腿之间的相对位置和速度
        #         delta_p_ir_r = p_1r_r - p_2r_r
        #         delta_v_ir_r = v_1r_r - v_2r_r

        #         # 计算S矩阵并加入到A矩阵中
        #         S_mat = -get_cross_matrix(delta_p_ir_r)
        #         A.append(S_mat)
        #         b.append(delta_v_ir_r.view(3, 1))  # 列向量

        #     A = torch.cat(A, dim=0).to(device)  # 确保A张量在同一设备
        #     b = torch.cat(b, dim=0).to(device)  # 确保b张量在同一设备

        #     # 使用torch.linalg.lstsq进行最小二乘求解
        #     result = torch.linalg.lstsq(A, b)
        #     w_pr_r = result.solution[:3].view(-1)  # 解向量
        #     return w_pr_r


        # w_pr_r = compute_relative_angular_velocity()
        # print("Relative angular velocity w_p/r^r:", w_pr_r.cpu().numpy().round(5))
        # print(f"True :{calculate_value([0, 0], 'w_pr_r').cpu().numpy().round(5)}")
        # # 计算相对角速度 w_pr_r=====================================================================================


        # # 计算 p_i/r_r =====================================================================
        
        # foot="FL"
        # # 关节索引
        # idx_hip = robot.data.body_names.index(f"{foot}_hip")
        # idx_thigh = robot.data.body_names.index(f"{foot}_thigh")
        # idx_calf = robot.data.body_names.index(f"{foot}_calf")
        # idx_foot = robot.data.body_names.index(f"{foot}_foot")
        # idx_root = robot.data.body_names.index("base")

        # # 连杆长度
        # p_hip = calculate_value([0, idx_hip], 'p_ir_r').cpu().numpy()
        # p_thigh = calculate_value([0, idx_thigh], 'p_ir_r').cpu().numpy()
        # p_calf = calculate_value([0, idx_calf], 'p_ir_r').cpu().numpy()
        # p_foot = calculate_value([0, idx_foot], 'p_ir_r').cpu().numpy()

        # L0 = np.linalg.norm(p_thigh - p_hip)
        # L1 = np.linalg.norm(p_calf - p_thigh)
        # L2 = np.linalg.norm(p_foot - p_calf)
        
        # def quat_to_rot_matrix(quat):
        #     w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            
        #     # 计算旋转矩阵
        #     R = torch.tensor([
        #         [1 - 2*y**2 - 2*z**2, 2*(x*y - w*z), 2*(x*z + w*y)],
        #         [2*(x*y + w*z), 1 - 2*x**2 - 2*z**2, 2*(y*z - w*x)],
        #         [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*x**2 - 2*y**2]
        #     ])
        #     return R

        # def get_T_i_j(i, j):
        #     R_i_j = math_utils.quat_mul(
        #         math_utils.quat_conjugate(robot.data.body_quat_w[0, j]),
        #         robot.data.body_quat_w[0, i]
        #     )
        #     p_ij_j = math_utils.quat_rotate_inverse(
        #         robot.data.body_quat_w[0, j],
        #         (robot.data.body_pos_w[0, i] - robot.data.body_pos_w[0, j])
        #     )
        #     T = torch.eye(4)
        #     T[:3, :3] = quat_to_rot_matrix(R_i_j)
        #     T[:3, 3] = p_ij_j
        #     return T
        # T_i_r = get_T_i_j(idx_hip, idx_root) @ get_T_i_j(idx_thigh, idx_hip) @ get_T_i_j(idx_calf, idx_thigh) @ get_T_i_j(idx_foot, idx_calf)
        # print(f"p_ir_r: {T_i_r[:3, 3].cpu().numpy().round(3)}")
        # p_ih_h_true = math_utils.quat_rotate_inverse(
        #     robot.data.body_quat_w[0, idx_hip],
        #     (robot.data.body_pos_w[0, idx_foot] - robot.data.body_pos_w[0, idx_hip])
        # )

        # p_ir_r_true = math_utils.quat_rotate_inverse(
        #     robot.data.body_quat_w[0, idx_root],
        #     (robot.data.body_pos_w[0, idx_foot] - robot.data.body_pos_w[0, idx_root])
        # )

        # print(f"P_ih_h_true: {p_ih_h_true.cpu().numpy().round(3)}")
        # print(f"p_ir_r_true: {p_ir_r_true.cpu().numpy().round(3)}")
        
        # 计算 p_i/r_r =====================================================================


        # GMY =========================================================================================================


        


        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 关闭模拟环境
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
