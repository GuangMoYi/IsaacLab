# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse                                  # 用于解析命令行参数
import sys                                       # 用于访问与 Hydra（Python 解释器）相关的变量和函数
from isaaclab.app import AppLauncher             # 用于启动 Isaac Sim 应用程序，用于初始化模拟环境
import cli_args                                  # 用于处理命令行参数，跳过 isort（代码格式化工具）的排序

# 添加argparse参数
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")   # 创建命令行参数解析器
    # 是否在训练过程中录制视频
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")    
    # 录制视频的时长（训练步数）        
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")  
    # 视频录制的间隔       
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
    # 模拟的环境数量
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
    # 强化学习任务名称，查找任务终端输入： ./isaaclab.sh -p scripts/environments/list_envs.py
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-X30-v0", help="Name of the task.")
    # 随机种子
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    # 最大迭代次数
parser.add_argument("--max_iterations", type=int, default=10000, help="RL Policy training iterations.")

# 添加 RSL-RL 相关的超参数（例如学习率、折扣因子等）
cli_args.add_rsl_rl_args(parser)
# 添加 Isaac Sim 相关的参数（例如 GPU 设备选择等）
AppLauncher.add_app_launcher_args(parser)
# 存储用户提供的命令行参数；存储 Hydra 框架的额外参数
args_cli, hydra_args = parser.parse_known_args()

# --video 为 True 时，开启相机
if args_cli.video:
    args_cli.enable_cameras = True

# 清理 sys.argv 以便 Hydra 解析配置文件
sys.argv = [sys.argv[0]] + hydra_args

# 创建 AppLauncher 实例，启动 Isaac Sim 模拟器 并获取 simulation_app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# 导入强化学习和环境相关库
import gymnasium as gym                                                         # 用于创建和管理RL环境
import os                                                                       # 用于处理文件和目录路径
import torch                                                                    # 用于张量计算和深度学习
from datetime import datetime                                                   # 用于获取当前时间（用于命名日志文件）

from rsl_rl.runners import OnPolicyRunner                                       # RSL-RL训练的核心类，负责on-policy的RL

from isaaclab.envs import (
    DirectMARLEnv,                                                              # 多智能体环境
    DirectMARLEnvCfg,                                                           # 多智能体的配置类
    DirectRLEnvCfg,                                                             # 单智能体的配置类
    ManagerBasedRLEnvCfg,                                                       # 基于管理器的单智能体环境的配置类
    multi_agent_to_single_agent,                                                # 多智能体环境转换为单智能体
)
from isaaclab.utils.dict import print_dict                                      # 打印字典
from isaaclab.utils.io import dump_pickle, dump_yaml                            # 保存环境和智能体的参数

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper       # RSL-RL的配置类和环境包装器

import isaaclab_tasks                                                           # RL任务集
from isaaclab_tasks.utils import get_checkpoint_path                            # 获取模型检查点路径（用于恢复训练）
from isaaclab_tasks.utils.hydra import hydra_task_config                        # Hydra 任务配置管理

# 基于 Hydra 解析 args_cli.task 任务的 默认配置
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""

    # 使用 CLI 传入的参数更新 Hydra 配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # 动态调整环境数量和最大迭代次数
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # 设置随机种子并选择计算设备
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 设定日志目录为本项目的根目录IsaacLab下的/logs/rsl_rl/{experiment_name}
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        # 转化为绝对路径
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
        # 获取当前时间并转化为字符串
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
        # 将 log_dir 和 log_root_path 拼接，得到最终的存储路径
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # RL训练： 创建 Isaac Gym 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
        # 如果有要求，将多智能体环境转换为单智能体环境
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # RL测试： 从指定的检查点加载模型
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 录制视频
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # 封装 Gym 环境，使其适配 RSL-RL 框架；动作裁剪 以防止超出合法范围
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # 创建 RSL-RL 训练 Runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # 记录 Git 代码状态（用于实验可复现）
    runner.add_git_repo_to_log(__file__)

    # RL测试： 加载之前训练的模型参数
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        runner.load(resume_path)

    # 保存训练参数： YAML 和 Pickle两种形式
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg) 
    
    # 开始RL训练
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    # 关闭仿真环境
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
