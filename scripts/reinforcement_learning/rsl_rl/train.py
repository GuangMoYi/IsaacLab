# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

# 导入必要的模块
import argparse  # 用于命令行参数解析
import sys  # 系统相关功能

from isaaclab.app import AppLauncher  # Isaac Lab应用启动器

# 本地导入
import cli_args  # isort: skip  # 命令行参数相关模块

# 添加argparse参数 - 定义训练脚本的命令行参数
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")  # 是否录制训练视频
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")  # 录制视频长度（步数）
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")  # 视频录制间隔（步数）
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")  # 并行环境数量
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-X30-v0", help="Name of the task.")  # 任务名称
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)  # RL智能体配置入口点名称
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")  # 环境随机种子
parser.add_argument("--max_iterations", type=int, default=50000, help="RL Policy training iterations.")  # 最大训练迭代次数
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)  # 是否使用分布式训练（多GPU或多节点）
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")  # 是否导出IO描述符
# 添加RSL-RL命令行参数
cli_args.add_rsl_rl_args(parser)
# 添加AppLauncher命令行参数
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()  # 解析命令行参数

# 如果需要录制视频，总是启用摄像头
if args_cli.video:
    args_cli.enable_cameras = True

# 为Hydra清理sys.argv
sys.argv = [sys.argv[0]] + hydra_args

# 启动omniverse应用
app_launcher = AppLauncher(args_cli)  # 创建应用启动器
simulation_app = app_launcher.app  # 获取仿真应用实例

"""Check for minimum supported RSL-RL version."""

# 导入版本检查和平台相关模块
import importlib.metadata as metadata  # 用于获取已安装包的元数据
import platform  # 用于获取平台信息

from packaging import version  # 用于版本比较

# 检查最低支持的rsl-rl版本
RSL_RL_VERSION = "3.0.1"  # 最低要求的RSL-RL版本
installed_version = metadata.version("rsl-rl-lib")  # 获取当前安装的版本
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):  # 如果当前版本低于要求版本
    # 根据操作系统选择安装命令
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    # 打印版本不匹配的错误信息
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)  # 退出程序

"""Rest everything follows."""

# 导入强化学习相关模块
import gymnasium as gym  # 强化学习环境接口
import os  # 操作系统接口
import torch  # PyTorch深度学习框架
from datetime import datetime  # 日期时间处理

import omni  # Omniverse核心模块
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # RSL-RL训练器

# 导入Isaac Lab环境相关模块
from isaaclab.envs import (
    DirectMARLEnv,  # 直接多智能体强化学习环境
    DirectMARLEnvCfg,  # 直接多智能体环境配置
    DirectRLEnvCfg,  # 直接强化学习环境配置
    ManagerBasedRLEnvCfg,  # 基于管理器的强化学习环境配置
    multi_agent_to_single_agent,  # 多智能体转单智能体函数
)
from isaaclab.utils.dict import print_dict  # 字典打印工具
from isaaclab.utils.io import dump_pickle, dump_yaml  # 数据序列化工具

# 导入RSL-RL相关模块
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401  # Isaac Lab任务模块
from isaaclab_tasks.utils import get_checkpoint_path  # 获取检查点路径
from isaaclab_tasks.utils.hydra import hydra_task_config  # Hydra任务配置

# PLACEHOLDER: Extension template (do not remove this comment)

# 配置PyTorch的CUDA后端设置
torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32矩阵乘法
torch.backends.cudnn.allow_tf32 = True  # 允许TF32卷积
torch.backends.cudnn.deterministic = False  # 非确定性算法（提高性能）
torch.backends.cudnn.benchmark = False  # 禁用CUDNN基准测试


@hydra_task_config(args_cli.task, args_cli.agent)  # 使用Hydra装饰器配置任务和智能体
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""  # 使用RSL-RL智能体进行训练
    # 使用非Hydra命令行参数覆盖配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)  # 更新RSL-RL配置
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs  # 设置环境数量
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )  # 设置最大迭代次数

    # 设置环境随机种子
    # 注意：某些随机化发生在环境初始化过程中，所以我们在这里设置种子
    env_cfg.seed = agent_cfg.seed  # 设置环境种子
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device  # 设置仿真设备

    # 多GPU训练配置
    if args_cli.distributed:  # 如果启用分布式训练
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"  # 设置环境设备为当前GPU
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"  # 设置智能体设备为当前GPU

        # 设置种子以在不同线程中保持多样性
        seed = agent_cfg.seed + app_launcher.local_rank  # 为每个进程设置不同的种子
        env_cfg.seed = seed  # 设置环境种子
        agent_cfg.seed = seed  # 设置智能体种子

    # 指定实验日志目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)  # 构建日志根路径
    log_root_path = os.path.abspath(log_root_path)  # 获取绝对路径
    print(f"[INFO] Logging experiment in directory: {log_root_path}")  # 打印日志目录信息
    # 指定运行日志目录：{时间戳}_{运行名称}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 生成时间戳格式的目录名
    # Ray Tune工作流使用下面的日志行提取实验名称，因此不要更改它（参见PR #2346, comment-2819298849）
    print(f"Exact experiment name requested from command line: {log_dir}")  # 打印确切的实验名称
    if agent_cfg.run_name:  # 如果有运行名称
        log_dir += f"_{agent_cfg.run_name}"  # 添加运行名称到目录名
    log_dir = os.path.join(log_root_path, log_dir)  # 构建完整的日志目录路径

    # 如果请求，设置IO描述符导出标志
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):  # 如果是基于管理器的RL环境
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors  # 设置IO描述符导出标志
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )  # 警告：IO描述符仅支持基于管理器的RL环境

    # 为环境设置日志目录（适用于所有环境类型）
    env_cfg.log_dir = log_dir  # 设置环境日志目录

    # 创建Isaac环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)  # 创建强化学习环境

    # 如果RL算法需要，转换为单智能体实例
    if isinstance(env.unwrapped, DirectMARLEnv):  # 如果是直接多智能体环境
        env = multi_agent_to_single_agent(env)  # 转换为单智能体环境

    # 在创建新日志目录之前保存恢复路径
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":  # 如果需要恢复或使用蒸馏算法
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)  # 获取检查点路径

    # 为视频录制包装环境
    if args_cli.video:  # 如果需要录制视频
        video_kwargs = {  # 视频录制参数
            "video_folder": os.path.join(log_dir, "videos", "train"),  # 视频保存文件夹
            "step_trigger": lambda step: step % args_cli.video_interval == 0,  # 视频录制触发条件
            "video_length": args_cli.video_length,  # 视频长度
            "disable_logger": True,  # 禁用日志记录器
        }
        print("[INFO] Recording videos during training.")  # 打印视频录制信息
        print_dict(video_kwargs, nesting=4)  # 打印视频参数
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # 包装环境以录制视频

    # 为RSL-RL包装环境
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # 使用RSL-RL向量环境包装器
    
    # GMY: 将 num_steps_per_env 传递给环境对象，供课程学习使用
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '__class__'):
        env.unwrapped._num_steps_per_env = agent_cfg.num_steps_per_env

    # 从RSL-RL创建训练器    
    if agent_cfg.class_name == "OnPolicyRunner":  # 如果是策略梯度训练器
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)  # 创建策略梯度训练器
    elif agent_cfg.class_name == "DistillationRunner":  # 如果是蒸馏训练器
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)  # 创建蒸馏训练器
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")  # 抛出不支持训练器类型的错误
    # 将git状态写入日志
    runner.add_git_repo_to_log(__file__)  # 添加git仓库信息到日志
    # 加载检查点
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":  # 如果需要恢复或使用蒸馏算法
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")  # 打印加载检查点信息
        # 加载之前训练的模型
        runner.load(resume_path)  # 加载检查点

    # 将配置转储到日志目录
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)  # 保存环境配置为YAML
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)  # 保存智能体配置为YAML
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)  # 保存环境配置为pickle
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)  # 保存智能体配置为pickle

    # 运行训练
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)  # 开始学习过程

    # 关闭仿真器
    env.close()  # 关闭环境


if __name__ == "__main__":
    # 运行主函数
    main()  # 调用主函数
    # 关闭仿真应用
    simulation_app.close()  # 关闭仿真应用
