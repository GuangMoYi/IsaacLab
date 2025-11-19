# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


# GMY: 课程学习 - 逐步增加跟随平台奖励的权重
def platform_following_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    initial_iterations: int = 0,  # 前N代不启用跟随奖励
    final_weight_orientation: float = 10.0,  # 姿态对齐的最终权重
    final_weight_angular_velocity: float = 30.0,  # 角速度同步的最终权重（通常需要更大，因为角速度误差更难控制）
    ramp_iterations: int = 100,  # 用N代逐渐增加到最终权重
    num_steps_per_env: int | None = None,  # 每环境每代的步数（如果为None，则自动从环境对象读取）
) -> float:
    """课程学习：逐步增加跟随平台奖励的权重。
    
    策略：
    1. 前 initial_iterations 代：权重为0，让机器狗先学会在平台上正常走路
    2. 接下来 ramp_iterations 代：权重从0线性增加到 final_weight
    3. 之后：权重保持为 final_weight
    
    注意：姿态对齐和角速度同步使用不同的权重，因为角速度误差通常更难控制。
    
    Args:
        env: 环境实例
        env_ids: 环境ID（未使用，但需要符合接口）
        initial_iterations: 初始阶段代数（权重为0）
        final_weight_orientation: 姿态对齐的最终权重
        final_weight_angular_velocity: 角速度同步的最终权重（通常需要更大）
        ramp_iterations: 权重增加的代数
        num_steps_per_env: 每环境每代的步数。如果为None，则尝试从环境对象中读取。
            如果环境对象没有这个属性，则使用默认值24。
    
    Returns:
        角速度同步的权重值（用于日志显示）
    """
    # 从环境配置中获取环境数量
    num_envs = env.cfg.scene.num_envs
    
    # 自动获取 num_steps_per_env
    if num_steps_per_env is None:
        # 尝试从环境对象中读取（如果训练时已经设置）
        if hasattr(env, '_num_steps_per_env') and env._num_steps_per_env is not None:
            num_steps_per_env = env._num_steps_per_env
        else:
            # 使用默认值（常见的RSL-RL配置值）
            num_steps_per_env = 24
            # 将值存储到环境对象中，以便后续使用
            env._num_steps_per_env = num_steps_per_env
    
    # 计算当前代数：从步数推断代数
    # 每代的步数 = num_steps_per_env * num_envs
    steps_per_iteration = num_steps_per_env * num_envs
    current_step = env.common_step_counter
    current_iteration = current_step // steps_per_iteration if steps_per_iteration > 0 else 0
    
    # 第一阶段：权重为0（让机器狗先学会走路）
    if current_iteration < initial_iterations:
        weight_orientation = 0.0
        weight_angular_velocity = 0.0
    # 第二阶段：线性增加权重
    elif current_iteration < initial_iterations + ramp_iterations:
        progress = (current_iteration - initial_iterations) / ramp_iterations
        weight_orientation = final_weight_orientation * progress
        weight_angular_velocity = final_weight_angular_velocity * progress
    # 第三阶段：保持最终权重
    else:
        weight_orientation = final_weight_orientation
        weight_angular_velocity = final_weight_angular_velocity
    
    # 调试输出（每1000代输出一次）
    if not hasattr(env, '_last_curriculum_debug_iteration'):
        env._last_curriculum_debug_iteration = -1
    if current_iteration != env._last_curriculum_debug_iteration and current_iteration % 1000 == 0:
        print(f"[课程学习调试] 当前代数={current_iteration}, 当前步数={current_step}, 每代步数={steps_per_iteration}")
        print(f"  姿态对齐权重={weight_orientation:.4f}, 角速度同步权重={weight_angular_velocity:.4f}")
        env._last_curriculum_debug_iteration = current_iteration
    
    # 更新两个跟随平台奖励的权重（使用不同的权重）
    # 基座姿态对齐奖励
    if "base_platform_parallel" in env.reward_manager._term_names:
        term_cfg = env.reward_manager.get_term_cfg("base_platform_parallel")
        term_cfg.weight = weight_orientation
        env.reward_manager.set_term_cfg("base_platform_parallel", term_cfg)
    
    # 相对角速度同步奖励（使用更大的权重）
    if "relative_angular_velocity_tracking" in env.reward_manager._term_names:
        term_cfg = env.reward_manager.get_term_cfg("relative_angular_velocity_tracking")
        term_cfg.weight = weight_angular_velocity
        env.reward_manager.set_term_cfg("relative_angular_velocity_tracking", term_cfg)
    
    # 返回角速度同步的权重（用于日志显示）
    return weight_angular_velocity
