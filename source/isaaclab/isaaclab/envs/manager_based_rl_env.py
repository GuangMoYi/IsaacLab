# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from isaacsim.core.version import get_version

from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.ui.widgets import ManagerLiveVisualizer

from .common import VecEnvStepReturn
from .manager_based_env import ManagerBasedEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


class ManagerBasedRLEnv(ManagerBasedEnv, gym.Env):
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # -- counter for curriculum
        self.common_step_counter = 0

        # initialize the episode length buffer BEFORE loading the managers to use it in mdp functions.
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)

        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)
        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        print("[INFO]: Completed setting up the environment...")

    """
    Properties.
    """

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()
        
        # -- initialize platform predictor (neural network for learning platform motion)
        # 在环境初始化时初始化预测器，这样在第一次step时就可以使用预测结果
        self._init_platform_predictor()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- update platform history (before reward computation so rewards can use it)
        self._update_platform_history()
        # -- update platform predictor (train on previous step's data, then predict for current step)
        self._update_platform_predictor()
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- update curriculum
        self.curriculum_manager.compute()
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        
        # GMY: 定期打印相对静止评估指标到终端
        self._print_relative_stationary_metrics()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return a numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            # call the parent class to close the environment
            super().close()

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
                term_dict = {}
                for term_name, term_dim, term_cfg in zip(group_term_names, group_dim, group_term_cfgs):
                    low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                    high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                    term_dict[term_name] = gym.spaces.Box(low=low, high=high, shape=term_dim)
                self.single_observation_space[group_name] = gym.spaces.Dict(term_dict)
        # action space (unbounded since we don't impose any limits)
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- reset platform history for reset environments
        self._reset_platform_history(env_ids)
        
        # -- reset platform predictor for reset environments
        # 如果所有环境都重置，清空预测器的状态
        if len(env_ids) == self.num_envs:
            if hasattr(self, '_platform_predictor'):
                # 清空预测器的训练缓冲区
                if hasattr(self._platform_predictor, 'training_buffer'):
                    self._platform_predictor.training_buffer['inputs'].clear()
                    self._platform_predictor.training_buffer['targets'].clear()
                # 重置预测状态
                self._last_prediction = None
                self._last_prediction_step = -1
        # 确保预测器已初始化
        self._init_platform_predictor()

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

    def _update_platform_history(self):
        """更新平台历史信息（姿态、角速度等）。
        
        在每一步中调用此函数来记录平台的当前状态和历史信息。
        这些信息可以在奖励函数中使用，而不需要增加观测维度。
        """
        # 检查场景中是否有平台
        try:
            platform = self.scene["platform"]
        except KeyError:
            # 如果场景中没有平台，直接返回
            return
        
        # 初始化平台历史信息存储（如果还没有初始化）
        if not hasattr(self, '_platform_history'):
            self._platform_history = {
                'quat_w': [],  # 姿态四元数历史（世界坐标系）
                'ang_vel_w': [],  # 角速度历史（世界坐标系）
                'lin_vel_w': [],  # 线速度历史（世界坐标系）
                'pos_w': [],  # 位置历史（世界坐标系）
                'max_history_length': 150,  # 增加最大历史长度：从100增加到150，确保有足够的历史数据用于预测（需要delay_steps+history_length）
            }
        
        # 获取当前平台状态
        current_quat = platform.data.root_quat_w.clone()
        current_ang_vel = platform.data.root_ang_vel_w.clone()
        current_lin_vel = platform.data.root_lin_vel_w.clone()
        current_pos = platform.data.root_pos_w.clone()
        
        # 添加到历史记录（使用循环缓冲区）
        self._platform_history['quat_w'].append(current_quat)
        self._platform_history['ang_vel_w'].append(current_ang_vel)
        self._platform_history['lin_vel_w'].append(current_lin_vel)
        self._platform_history['pos_w'].append(current_pos)
        
        # 限制历史长度
        max_len = self._platform_history['max_history_length']
        if len(self._platform_history['quat_w']) > max_len:
            self._platform_history['quat_w'].pop(0)
            self._platform_history['ang_vel_w'].pop(0)
            self._platform_history['lin_vel_w'].pop(0)
            self._platform_history['pos_w'].pop(0)
    
    def _reset_platform_history(self, env_ids: Sequence[int]):
        """重置指定环境的平台历史信息。
        
        Args:
            env_ids: 需要重置的环境ID列表
        """
        # 如果平台历史信息还没有初始化，则不需要重置
        if not hasattr(self, '_platform_history'):
            return
        
        # 对于重置的环境，清空历史记录（因为环境已经重置）
        # 注意：由于所有环境共享同一个平台历史，我们不清空整个历史
        # 而是在需要时根据env_ids过滤
        # 这里我们只清空历史，让它在下一步重新开始记录
        if len(env_ids) == self.num_envs:
            # 如果所有环境都重置，清空历史
            self._platform_history['quat_w'].clear()
            self._platform_history['ang_vel_w'].clear()
            self._platform_history['lin_vel_w'].clear()
            self._platform_history['pos_w'].clear()
        
        # 重置机器人角速度历史（用于趋势分析）
        if hasattr(self, '_robot_ang_vel_history'):
            # 清空机器人角速度历史（因为环境重置了）
            if len(env_ids) == self.num_envs:
                # 如果所有环境都重置，清空历史
                self._robot_ang_vel_history.clear()
            else:
                # 如果只是部分环境重置，我们无法精确处理，所以清空整个历史
                # 这是合理的，因为历史记录是全局的
                self._robot_ang_vel_history.clear()
        
        # 重置预测器状态（不清空网络参数，只清空预测缓存）
        if hasattr(self, '_last_prediction'):
            self._last_prediction = None
            self._last_prediction_step = -1
    
    def get_platform_history(self, history_length: int = None) -> dict[str, torch.Tensor]:
        """获取平台历史信息。
        
        Args:
            history_length: 要获取的历史长度。如果为None，则返回所有历史。
        
        Returns:
            包含平台历史信息的字典：
            - 'quat_w': 姿态四元数历史 [history_length, num_envs, 4]
            - 'ang_vel_w': 角速度历史 [history_length, num_envs, 3]
            - 'lin_vel_w': 线速度历史 [history_length, num_envs, 3]
            - 'pos_w': 位置历史 [history_length, num_envs, 3]
            - 'current_quat_w': 当前姿态 [num_envs, 4]
            - 'current_ang_vel_w': 当前角速度 [num_envs, 3]
            - 'current_lin_vel_w': 当前线速度 [num_envs, 3]
            - 'current_pos_w': 当前位置 [num_envs, 3]
        """
        if not hasattr(self, '_platform_history') or len(self._platform_history['quat_w']) == 0:
            # 如果没有历史信息，返回当前状态
            try:
                platform = self.scene["platform"]
            except KeyError:
                return {}
            return {
                'current_quat_w': platform.data.root_quat_w,
                'current_ang_vel_w': platform.data.root_ang_vel_w,
                'current_lin_vel_w': platform.data.root_lin_vel_w,
                'current_pos_w': platform.data.root_pos_w,
            }
        
        # 获取历史记录
        quat_history = self._platform_history['quat_w']
        ang_vel_history = self._platform_history['ang_vel_w']
        lin_vel_history = self._platform_history['lin_vel_w']
        pos_history = self._platform_history['pos_w']
        
        # 限制历史长度
        if history_length is not None:
            quat_history = quat_history[-history_length:]
            ang_vel_history = ang_vel_history[-history_length:]
            lin_vel_history = lin_vel_history[-history_length:]
            pos_history = pos_history[-history_length:]
        
        # 堆叠历史记录
        result = {
            'quat_w': torch.stack(quat_history, dim=0) if len(quat_history) > 0 else None,
            'ang_vel_w': torch.stack(ang_vel_history, dim=0) if len(ang_vel_history) > 0 else None,
            'lin_vel_w': torch.stack(lin_vel_history, dim=0) if len(lin_vel_history) > 0 else None,
            'pos_w': torch.stack(pos_history, dim=0) if len(pos_history) > 0 else None,
        }
        
        # 添加当前状态
        try:
            platform = self.scene["platform"]
            result['current_quat_w'] = platform.data.root_quat_w
            result['current_ang_vel_w'] = platform.data.root_ang_vel_w
            result['current_lin_vel_w'] = platform.data.root_lin_vel_w
            result['current_pos_w'] = platform.data.root_pos_w
        except KeyError:
            pass  # 如果场景中没有平台，跳过添加当前状态
        
        return result
    
    def _init_platform_predictor(self):
        """初始化平台运动预测器（神经网络）"""
        try:
            platform = self.scene["platform"]
        except KeyError:
            return
        
        if not hasattr(self, '_platform_predictor'):
            from isaaclab_tasks.manager_based.locomotion.velocity.mdp.platform_predictor import PlatformMotionPredictor
            
            # 初始化预测器（提高精度：更大的网络和更长的历史）
            self._platform_predictor = PlatformMotionPredictor(
                history_length=30,  # 增加历史长度：从20增加到30，提供更多上下文信息
                hidden_size=128,    # 增加隐藏层大小：从64增加到128，提高模型容量
                num_layers=3,       # 增加LSTM层数：从2增加到3，增强时序建模能力
                prediction_horizon=0.2,
                learning_rate=5e-4, # 降低学习率：从1e-3降到5e-4，更稳定的训练
                device=self.device
            )
            self._platform_predictor.to(self.device)
            # 确保所有参数需要梯度
            for param in self._platform_predictor.parameters():
                param.requires_grad = True
            
            # 存储上一步的预测值（用于训练）
            self._last_prediction = None
            self._last_prediction_step = -1
    
    def _update_platform_predictor(self):
        """更新平台预测器：训练网络（使用延迟历史数据预测当前状态）
        
        训练逻辑：使用t-5之前的历史数据预测当前状态，与观测和奖励函数保持一致
        """
        try:
            platform = self.scene["platform"]
        except KeyError:
            return
        
        # 初始化预测器（如果还没有）
        if not hasattr(self, '_platform_predictor'):
            self._init_platform_predictor()
            return
        
        # 获取平台历史数据（需要足够长的历史）
        platform_history = self.get_platform_history(history_length=None)
        
        if platform_history.get('quat_w') is None or platform_history.get('ang_vel_w') is None:
            return
        
        delay_steps = 5  # 与观测和奖励函数保持一致
        
        # 如果历史数据足够（至少需要delay_steps + 2步），进行训练
        if len(platform_history['quat_w']) >= delay_steps + 2:
            from isaaclab.utils.math import euler_xyz_from_quat
            
            # 获取当前实际状态（作为训练目标）
            current_quat = platform.data.root_quat_w
            current_ang_vel = platform.data.root_ang_vel_w
            
            current_roll, current_pitch, _ = euler_xyz_from_quat(current_quat)
            current_roll_ang_vel = current_ang_vel[:, 0]
            current_pitch_ang_vel = current_ang_vel[:, 1]
            
            # 使用t-delay_steps之前的历史数据（不包括当前步）进行训练
            # 训练目标：使用延迟历史数据预测当前状态
            # 这样与观测和奖励函数使用的预测方法完全一致
            # 注意：我们需要使用延迟历史数据，模拟"在t-delay_steps时刻，预测当前时刻"的场景
            total_history_length = len(platform_history['quat_w'])
            if total_history_length >= delay_steps + 1:
                # 使用t-delay_steps之前的历史数据（不包括当前步）
                cutoff_idx = total_history_length - delay_steps - 1  # -1是因为不包括当前步
                if cutoff_idx > 0:
                    delayed_history = {
                        'quat_w': platform_history['quat_w'][:cutoff_idx],
                        'ang_vel_w': platform_history['ang_vel_w'][:cutoff_idx],
                    }
                    
                    # 更新网络（在线学习）
                    # 训练：使用延迟历史数据预测当前状态
                    self._platform_predictor.update(
                        delayed_history,
                        current_roll,
                        current_pitch,
                        current_roll_ang_vel,
                        current_pitch_ang_vel
                    )
    
    def get_platform_prediction(self) -> dict[str, torch.Tensor] | None:
        """获取平台运动预测结果
        
        Returns:
            预测结果字典，包含：
            - 'roll': 预测的roll角度 [num_envs]
            - 'pitch': 预测的pitch角度 [num_envs]
            - 'roll_ang_vel': 预测的roll角速度 [num_envs]
            - 'pitch_ang_vel': 预测的pitch角速度 [num_envs]
            如果预测器未初始化或历史数据不足，返回None
        """
        if not hasattr(self, '_platform_predictor') or self._last_prediction is None:
            return None
        return self._last_prediction
    
    def get_platform_prediction_for_observation(self, delay_steps: int = 5) -> dict[str, torch.Tensor] | None:
        """获取用于观测的平台预测结果（使用延迟历史数据预测当前时刻）
        
        这个方法专门用于观测空间：机器狗能看到t-delay_steps之前的数据，
        然后预测当前时刻（t时刻）的平台状态
        
        Args:
            delay_steps: 延迟步数，使用t-delay_steps之前的数据来预测t时刻
        
        Returns:
            预测结果字典，包含：
            - 'roll': 预测的当前roll角度 [num_envs]
            - 'pitch': 预测的当前pitch角度 [num_envs]
            - 'roll_ang_vel': 预测的当前roll角速度 [num_envs]
            - 'pitch_ang_vel': 预测的当前pitch角速度 [num_envs]
            如果预测器未初始化或历史数据不足，返回None
        """
        if not hasattr(self, '_platform_predictor'):
            return None
        
        # 获取平台历史数据（需要足够长的历史）
        platform_history = self.get_platform_history(history_length=None)
        
        if platform_history.get('quat_w') is None or platform_history.get('ang_vel_w') is None:
            return None
        
        # 使用延迟历史数据预测当前时刻
        predicted_roll, predicted_pitch, predicted_roll_ang_vel, predicted_pitch_ang_vel = \
            self._platform_predictor.predict_current_from_delayed_history(
                platform_history, delay_steps=delay_steps
            )
        
        return {
            'roll': predicted_roll,
            'pitch': predicted_pitch,
            'roll_ang_vel': predicted_roll_ang_vel,
            'pitch_ang_vel': predicted_pitch_ang_vel
        }
    
    def get_platform_delayed_history(self, delay_steps: int = 5, history_length: int = 10) -> dict[str, torch.Tensor] | None:
        """获取平台延迟历史数据（t-delay_steps之前的数据）
        
        用于观测空间：机器狗能看到t-delay_steps之前的历史数据
        
        Args:
            delay_steps: 延迟步数，返回t-delay_steps之前的数据
            history_length: 返回的历史长度
        
        Returns:
            历史数据字典，包含：
            - 'roll': 历史roll角度 [num_envs, history_length]
            - 'pitch': 历史pitch角度 [num_envs, history_length]
            - 'roll_ang_vel': 历史roll角速度 [num_envs, history_length]
            - 'pitch_ang_vel': 历史pitch角速度 [num_envs, history_length]
            如果历史数据不足，返回None
        """
        from isaaclab.utils.math import euler_xyz_from_quat
        
        platform_history = self.get_platform_history(history_length=None)
        
        if platform_history.get('quat_w') is None or platform_history.get('ang_vel_w') is None:
            return None
        
        quat_history = platform_history['quat_w']
        ang_vel_history = platform_history['ang_vel_w']
        
        # 处理列表或tensor格式
        if isinstance(quat_history, list):
            quat_history = torch.stack(quat_history, dim=0)
        if isinstance(ang_vel_history, list):
            ang_vel_history = torch.stack(ang_vel_history, dim=0)
        
        total_history_length = quat_history.shape[0]
        if total_history_length < delay_steps + 1:
            return None
        
        # 使用t-delay_steps之前的数据
        cutoff_idx = total_history_length - delay_steps
        if cutoff_idx <= 0:
            cutoff_idx = 1
        
        delayed_quat = quat_history[:cutoff_idx]
        delayed_ang_vel = ang_vel_history[:cutoff_idx]
        
        # 限制历史长度
        actual_history_length = min(history_length, delayed_quat.shape[0])
        if actual_history_length < 1:
            return None
        
        recent_quat = delayed_quat[-actual_history_length:]
        recent_ang_vel = delayed_ang_vel[-actual_history_length:]
        
        num_envs = recent_quat.shape[1]
        
        # 提取roll和pitch角度
        history_roll = torch.zeros(actual_history_length, num_envs, device=recent_quat.device)
        history_pitch = torch.zeros(actual_history_length, num_envs, device=recent_quat.device)
        
        for i in range(actual_history_length):
            roll_i, pitch_i, _ = euler_xyz_from_quat(recent_quat[i])
            history_roll[i] = roll_i
            history_pitch[i] = pitch_i
        
        # 提取roll和pitch角速度
        history_roll_ang_vel = recent_ang_vel[:, :, 0]
        history_pitch_ang_vel = recent_ang_vel[:, :, 1]
        
        # 转置为 [num_envs, actual_history_length]
        history_roll = history_roll.transpose(0, 1)
        history_pitch = history_pitch.transpose(0, 1)
        history_roll_ang_vel = history_roll_ang_vel.transpose(0, 1)
        history_pitch_ang_vel = history_pitch_ang_vel.transpose(0, 1)
        
        # 如果历史长度不足，用最后一个值填充
        if actual_history_length < history_length:
            padding_size = history_length - actual_history_length
            history_roll = torch.cat([
                history_roll,
                history_roll[:, -1:].expand(-1, padding_size)
            ], dim=1)
            history_pitch = torch.cat([
                history_pitch,
                history_pitch[:, -1:].expand(-1, padding_size)
            ], dim=1)
            history_roll_ang_vel = torch.cat([
                history_roll_ang_vel,
                history_roll_ang_vel[:, -1:].expand(-1, padding_size)
            ], dim=1)
            history_pitch_ang_vel = torch.cat([
                history_pitch_ang_vel,
                history_pitch_ang_vel[:, -1:].expand(-1, padding_size)
            ], dim=1)
        
        return {
            'roll': history_roll,
            'pitch': history_pitch,
            'roll_ang_vel': history_roll_ang_vel,
            'pitch_ang_vel': history_pitch_ang_vel
        }

    # GMY changed: 相对静止==打印指标
    def _print_relative_stationary_metrics(self):
        """定期打印相对静止评估指标到终端。
        
        每1000步打印一次，显示所有环境的统计信息：
        - 相对速度误差（m/s）
        - 基座与平台姿态误差（rad）
        - 相对位置漂移（m）
        """
        # 初始化打印计数器
        if not hasattr(self, '_metric_print_counter'):
            self._metric_print_counter = 0
        
        self._metric_print_counter += 1
        
        # 每1000步打印一次
        if self._metric_print_counter % 500 == 0:
            try:
                # 获取debug观测组中的指标（不更新历史）
                debug_obs = self.observation_manager.compute_group("debug", update_history=False)
                
                if debug_obs is not None and isinstance(debug_obs, dict):
                    # 获取指标值（计算所有环境的统计信息）
                    robot_metrics = {}  # 机器人相关指标
                    platform_metrics = {}  # 平台自身指标（用于对比）
                    
                    # 相对速度误差（机器人）
                    if "relative_velocity_error" in debug_obs:
                        rel_vel_error = debug_obs["relative_velocity_error"]
                        if isinstance(rel_vel_error, torch.Tensor):
                            robot_metrics["相对速度误差(m/s)"] = {
                                "mean": rel_vel_error.mean().item(),
                                "min": rel_vel_error.min().item(),
                                "max": rel_vel_error.max().item(),
                                "std": rel_vel_error.std().item(),
                            }
                    
                    # 基座姿态误差（机器人相对于平台）
                    if "base_platform_orientation_error" in debug_obs:
                        orientation_error = debug_obs["base_platform_orientation_error"]
                        if isinstance(orientation_error, torch.Tensor):
                            robot_metrics["基座姿态误差(rad)"] = {
                                "mean": orientation_error.mean().item(),
                                "min": orientation_error.min().item(),
                                "max": orientation_error.max().item(),
                                "std": orientation_error.std().item(),
                            }
                    
                    # 相对角速度误差（机器人相对于平台）
                    if "robot_relative_ang_vel_error" in debug_obs:
                        rel_ang_vel_error = debug_obs["robot_relative_ang_vel_error"]
                        if isinstance(rel_ang_vel_error, torch.Tensor):
                            robot_metrics["相对角速度误差(rad/s)"] = {
                                "mean": rel_ang_vel_error.mean().item(),
                                "min": rel_ang_vel_error.min().item(),
                                "max": rel_ang_vel_error.max().item(),
                                "std": rel_ang_vel_error.std().item(),
                            }
                    
                    # 平台自身姿态误差（用于对比）
                    if "platform_orientation_error" in debug_obs:
                        platform_orientation_error = debug_obs["platform_orientation_error"]
                        if isinstance(platform_orientation_error, torch.Tensor):
                            platform_metrics["平台姿态误差(rad)"] = {
                                "mean": platform_orientation_error.mean().item(),
                                "min": platform_orientation_error.min().item(),
                                "max": platform_orientation_error.max().item(),
                                "std": platform_orientation_error.std().item(),
                            }
                    
                    # 平台自身角速度误差（用于对比）
                    if "platform_ang_vel_error" in debug_obs:
                        platform_ang_vel_error = debug_obs["platform_ang_vel_error"]
                        if isinstance(platform_ang_vel_error, torch.Tensor):
                            platform_metrics["平台角速度误差(rad/s)"] = {
                                "mean": platform_ang_vel_error.mean().item(),
                                "min": platform_ang_vel_error.min().item(),
                                "max": platform_ang_vel_error.max().item(),
                                "std": platform_ang_vel_error.std().item(),
                            }
                    
                    # 机器狗运动指标
                    robot_motion_metrics = {}
                    
                    # 机器狗线速度大小
                    if "robot_lin_vel_magnitude" in debug_obs:
                        robot_lin_vel_mag = debug_obs["robot_lin_vel_magnitude"]
                        if isinstance(robot_lin_vel_mag, torch.Tensor):
                            # 如果是 [num_envs, 1] 形状，需要squeeze
                            if robot_lin_vel_mag.dim() > 1:
                                robot_lin_vel_mag = robot_lin_vel_mag.squeeze(-1)
                            robot_motion_metrics["机器狗线速度大小(m/s)"] = {
                                "mean": robot_lin_vel_mag.mean().item(),
                                "min": robot_lin_vel_mag.min().item(),
                                "max": robot_lin_vel_mag.max().item(),
                                "std": robot_lin_vel_mag.std().item(),
                            }
                    
                    # 机器狗角速度大小
                    if "robot_ang_vel_magnitude" in debug_obs:
                        robot_ang_vel_mag = debug_obs["robot_ang_vel_magnitude"]
                        if isinstance(robot_ang_vel_mag, torch.Tensor):
                            # 如果是 [num_envs, 1] 形状，需要squeeze
                            if robot_ang_vel_mag.dim() > 1:
                                robot_ang_vel_mag = robot_ang_vel_mag.squeeze(-1)
                            robot_motion_metrics["机器狗角速度大小(rad/s)"] = {
                                "mean": robot_ang_vel_mag.mean().item(),
                                "min": robot_ang_vel_mag.min().item(),
                                "max": robot_ang_vel_mag.max().item(),
                                "std": robot_ang_vel_mag.std().item(),
                            }
                    
                    # 机器狗线速度（世界坐标系，xyz分量）
                    if "robot_lin_vel_w" in debug_obs:
                        robot_lin_vel_w = debug_obs["robot_lin_vel_w"]
                        if isinstance(robot_lin_vel_w, torch.Tensor):
                            robot_motion_metrics["机器狗线速度X(m/s)"] = {
                                "mean": robot_lin_vel_w[:, 0].mean().item(),
                                "min": robot_lin_vel_w[:, 0].min().item(),
                                "max": robot_lin_vel_w[:, 0].max().item(),
                                "std": robot_lin_vel_w[:, 0].std().item(),
                            }
                            robot_motion_metrics["机器狗线速度Y(m/s)"] = {
                                "mean": robot_lin_vel_w[:, 1].mean().item(),
                                "min": robot_lin_vel_w[:, 1].min().item(),
                                "max": robot_lin_vel_w[:, 1].max().item(),
                                "std": robot_lin_vel_w[:, 1].std().item(),
                            }
                            robot_motion_metrics["机器狗线速度Z(m/s)"] = {
                                "mean": robot_lin_vel_w[:, 2].mean().item(),
                                "min": robot_lin_vel_w[:, 2].min().item(),
                                "max": robot_lin_vel_w[:, 2].max().item(),
                                "std": robot_lin_vel_w[:, 2].std().item(),
                            }
                    
                    # 机器狗角速度（世界坐标系，xyz分量）
                    if "robot_ang_vel_w" in debug_obs:
                        robot_ang_vel_w = debug_obs["robot_ang_vel_w"]
                        if isinstance(robot_ang_vel_w, torch.Tensor):
                            robot_motion_metrics["机器狗角速度X(rad/s)"] = {
                                "mean": robot_ang_vel_w[:, 0].mean().item(),
                                "min": robot_ang_vel_w[:, 0].min().item(),
                                "max": robot_ang_vel_w[:, 0].max().item(),
                                "std": robot_ang_vel_w[:, 0].std().item(),
                            }
                            robot_motion_metrics["机器狗角速度Y(rad/s)"] = {
                                "mean": robot_ang_vel_w[:, 1].mean().item(),
                                "min": robot_ang_vel_w[:, 1].min().item(),
                                "max": robot_ang_vel_w[:, 1].max().item(),
                                "std": robot_ang_vel_w[:, 1].std().item(),
                            }
                            robot_motion_metrics["机器狗角速度Z(rad/s)"] = {
                                "mean": robot_ang_vel_w[:, 2].mean().item(),
                                "min": robot_ang_vel_w[:, 2].min().item(),
                                "max": robot_ang_vel_w[:, 2].max().item(),
                                "std": robot_ang_vel_w[:, 2].std().item(),
                            }
                    
                    # 获取课程学习权重（如果存在）
                    curriculum_weight = None
                    if hasattr(self, 'curriculum_manager'):
                        curriculum_state = self.curriculum_manager._curriculum_state
                        if "platform_following_weight" in curriculum_state:
                            weight_val = curriculum_state["platform_following_weight"]
                            if weight_val is not None:
                                curriculum_weight = weight_val
                    
                    # 打印指标（显示所有环境的统计信息）
                    if robot_metrics or platform_metrics or robot_motion_metrics:
                        num_envs = self.cfg.scene.num_envs
                        print(f"\n[相对静止指标] Step {self.common_step_counter} (共 {num_envs} 个环境):")
                        
                        # 打印课程学习权重
                        if curriculum_weight is not None:
                            print(f"  【课程学习权重】: 跟随平台奖励权重 = {curriculum_weight:.4f}")
                        
                        # 打印机器狗运动指标
                        if robot_motion_metrics:
                            print("  【机器狗运动指标】:")
                            for name, stats in robot_motion_metrics.items():
                                print(f"    {name}: 平均值={stats['mean']:.6f}, 最小值={stats['min']:.6f}, 最大值={stats['max']:.6f}, 标准差={stats['std']:.6f}")
                        
                        # 打印机器人指标
                        if robot_metrics:
                            print("  【机器人相对静止指标】:")
                            for name, stats in robot_metrics.items():
                                print(f"    {name}: 平均值={stats['mean']:.6f}, 最小值={stats['min']:.6f}, 最大值={stats['max']:.6f}, 标准差={stats['std']:.6f}")
                        
                        # 打印平台指标（用于对比）
                        if platform_metrics:
                            print("  【平台自身指标（对比用，理想值接近0）】:")
                            for name, stats in platform_metrics.items():
                                print(f"    {name}: 平均值={stats['mean']:.6f}, 最小值={stats['min']:.6f}, 最大值={stats['max']:.6f}, 标准差={stats['std']:.6f}")
                        
                        # 计算并打印比值分析
                        if robot_metrics and platform_metrics:
                            print("  【比值分析（机器人指标 / 平台指标，<1表示优于平台，>1表示劣于平台）】:")
                            
                            # 基座姿态误差比值
                            if "基座姿态误差(rad)" in robot_metrics and "平台姿态误差(rad)" in platform_metrics:
                                robot_orientation_mean = robot_metrics["基座姿态误差(rad)"]["mean"]
                                platform_orientation_mean = platform_metrics["平台姿态误差(rad)"]["mean"]
                                if platform_orientation_mean > 1e-8:  # 避免除零
                                    orientation_ratio = robot_orientation_mean / platform_orientation_mean
                                    status = "✓ 优于平台" if orientation_ratio < 1.0 else "✗ 劣于平台"
                                    print(f"    基座姿态误差比值: {orientation_ratio:.4f} ({status})")
                            
                            # 相对角速度误差比值
                            if "相对角速度误差(rad/s)" in robot_metrics and "平台角速度误差(rad/s)" in platform_metrics:
                                robot_ang_vel_mean = robot_metrics["相对角速度误差(rad/s)"]["mean"]
                                platform_ang_vel_mean = platform_metrics["平台角速度误差(rad/s)"]["mean"]
                                if platform_ang_vel_mean > 1e-8:  # 避免除零
                                    ang_vel_ratio = robot_ang_vel_mean / platform_ang_vel_mean
                                    status = "✓ 优于平台" if ang_vel_ratio < 1.0 else "✗ 劣于平台"
                                    print(f"    相对角速度误差比值: {ang_vel_ratio:.4f} ({status})")
                        
                        print()
            except Exception as e:
                # 如果获取指标失败，不中断训练（静默失败）
                pass
