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
        
        # -- initialize training data recorder
        try:
            from isaaclab.envs.training_data_recorder import TrainingDataRecorder
            self.training_data_recorder = TrainingDataRecorder(self)
            print("[INFO] Training Data Recorder initialized")
        except Exception as e:
            print(f"[WARNING] Failed to initialize Training Data Recorder: {e}")
            self.training_data_recorder = None

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
        # -- update observation history (for platform motion inference)
        self._update_observation_history()
        # -- infer platform motion from observations (if inference is enabled)
        self._infer_platform_motion_from_observations()
        # -- update platform predictor (train on previous step's data, then predict for current step)
        # 关键优化：不是每步都更新，而是每隔一定步数才更新，减少计算开销
        if not hasattr(self, '_last_platform_predictor_update_step'):
            self._last_platform_predictor_update_step = -1
        
        current_step = getattr(self, '_sim_step_counter', 0)
        # 关键改进：每步都训练预测器，提高训练频率
        # 从每5步训练一次改为每步都训练，加快预测器学习速度
        if current_step - self._last_platform_predictor_update_step >= 1:
            self._last_platform_predictor_update_step = current_step
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
        
        # -- record training data
        if hasattr(self, 'training_data_recorder') and self.training_data_recorder is not None:
            self.training_data_recorder.record_step()

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
        
        # -- record episode end data
        if hasattr(self, 'training_data_recorder') and self.training_data_recorder is not None:
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            self.training_data_recorder.record_episode_end(env_ids_tensor)
        
        # -- reset platform predictor for reset environments
        # 关键改进：不要清空训练缓冲区，保留已学习的运动模式
        # 如果所有环境都重置，只重置预测状态，但保留训练数据
        # 这样即使换了新的运动模式，神经网络也能快速适应（因为有在线学习）
        if len(env_ids) == self.num_envs:
            if hasattr(self, '_platform_predictor'):
                # 不清空训练缓冲区，保留已学习的运动模式
                # 如果确实需要重新学习（比如完全不同的运动模式），可以手动清空
                # 但通常情况下，在线学习可以适应新的运动模式
                
                # 只重置预测状态
                self._last_prediction = None
                self._last_prediction_step = -1
                
                # 可选：如果训练缓冲区太大，可以保留最近的一部分
                # 但不清空全部，这样神经网络可以快速适应新的运动模式
                if hasattr(self._platform_predictor, 'training_buffer'):
                    buffer_size = len(self._platform_predictor.training_buffer['inputs'])
                    if buffer_size > 500:  # 如果缓冲区太大，只保留最近500个样本
                        keep_size = 500
                        self._platform_predictor.training_buffer['inputs'] = \
                            self._platform_predictor.training_buffer['inputs'][-keep_size:]
                        self._platform_predictor.training_buffer['targets'] = \
                            self._platform_predictor.training_buffer['targets'][-keep_size:]
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
        """初始化平台运动预测器（神经网络，使用机器狗观测历史）"""
        try:
            robot = self.scene["robot"]
        except KeyError:
            return
        
        if not hasattr(self, '_platform_predictor'):
            from isaaclab_tasks.manager_based.locomotion.velocity.mdp.platform_predictor import PlatformMotionPredictor
            
            # 获取关节数量
            num_joints = robot.num_joints
            num_actions = robot.num_joints  # 假设动作数量等于关节数量
            
            # 获取velocity_commands和height_scan的维度
            try:
                velocity_commands = self.command_manager.get_command("base_velocity")
                num_velocity_commands = velocity_commands.shape[1] if velocity_commands.dim() > 1 else 3
            except:
                num_velocity_commands = 3  # 默认：x, y, yaw
            
            try:
                height_scanner = self.scene.sensors.get("height_scanner")
                if height_scanner is not None:
                    num_height_scan_points = height_scanner.data.ray_hits_w.shape[1] if hasattr(height_scanner.data, 'ray_hits_w') else 0
                else:
                    num_height_scan_points = 0
            except:
                num_height_scan_points = 0  # 如果没有高度扫描传感器，设为0
            
            # 初始化预测器（使用机器狗观测历史）
            # 关键改进：增强网络结构，提高表达能力
            self._platform_predictor = PlatformMotionPredictor(
                history_length=50,  # 使用50步观测历史
                num_joints=num_joints,
                num_actions=num_actions,
                num_velocity_commands=num_velocity_commands,
                num_height_scan_points=num_height_scan_points,
                hidden_size=512,  # 大幅增加隐藏层大小：从256增加到512
                num_layers=4,  # 增加LSTM层数：从3增加到4
                prediction_horizon=0.1,
                prediction_steps=5,  # 预测未来5步（约0.1秒，dt=0.02s）
                learning_rate=3e-4,  # 降低学习率：从1e-3降低到3e-4，提高训练稳定性
                device=str(self.device)  # 确保是字符串格式
            )
            # 先移动主模型到设备
            self._platform_predictor.to(self.device)
            # 然后移动生产模型到设备（如果已创建）
            if hasattr(self._platform_predictor, 'production_model') and self._platform_predictor.production_model is not None:
                self._platform_predictor.production_model.to(self.device)
            # 确保所有参数需要梯度
            for param in self._platform_predictor.parameters():
                param.requires_grad = True
            
            # 存储上一步的预测值（用于训练）
            self._last_prediction = None
            self._last_prediction_step = -1
    
    def _init_platform_inference(self):
        """初始化平台运动推断器（从观测信息推断平台运动）"""
        try:
            robot = self.scene["robot"]
        except KeyError:
            return
        
        if not hasattr(self, '_platform_inference'):
            from isaaclab_tasks.manager_based.locomotion.velocity.mdp.platform_inference import PlatformMotionInference
            
            # 获取关节数量
            num_joints = robot.num_joints
            num_actions = robot.num_joints  # 假设动作数量等于关节数量
            
            # 获取velocity_commands和height_scan的维度
            try:
                velocity_commands = self.command_manager.get_command("base_velocity")
                num_velocity_commands = velocity_commands.shape[1] if velocity_commands.dim() > 1 else 3
            except:
                num_velocity_commands = 3  # 默认：x, y, yaw
            
            try:
                height_scanner = self.scene.sensors.get("height_scanner")
                if height_scanner is not None:
                    # height_scan的形状通常是 [num_envs, num_scan_points]
                    # 需要从传感器数据中获取实际维度
                    num_height_scan_points = height_scanner.data.ray_hits_w.shape[1] if hasattr(height_scanner.data, 'ray_hits_w') else 0
                else:
                    num_height_scan_points = 0
            except:
                num_height_scan_points = 0  # 如果没有高度扫描传感器，设为0
            
            # 初始化推断器（包含所有观测信息）
            self._platform_inference = PlatformMotionInference(
                history_length=50,  # 使用50步观测历史
                num_joints=num_joints,
                num_actions=num_actions,
                num_velocity_commands=num_velocity_commands,
                num_height_scan_points=num_height_scan_points,
                hidden_size=256,
                num_layers=3,
                learning_rate=1e-3,
                device=str(self.device)
            )
            self._platform_inference.to(self.device)
            
            # 初始化观测历史缓冲区（包含所有观测信息）
            self._observation_history = {
                'base_lin_vel': [],
                'base_ang_vel': [],
                'projected_gravity': [],
                'velocity_commands': [],
                'joint_pos': [],
                'joint_vel': [],
                'actions': [],
                'height_scan': [],
                'max_history_length': 50,
            }
            
            # 初始化推断的平台历史（用于预测器）
            self._inferred_platform_history = {
                'roll': [],
                'pitch': [],
                'roll_ang_vel': [],
                'pitch_ang_vel': [],
                'max_history_length': 150,
            }
    
    def _update_observation_history(self):
        """更新观测历史（用于平台运动推断）"""
        try:
            robot = self.scene["robot"]
        except KeyError:
            return
        
        # 初始化观测历史（如果还没有）
        if not hasattr(self, '_observation_history'):
            self._observation_history = {
                'base_lin_vel': [],
                'base_ang_vel': [],
                'projected_gravity': [],
                'velocity_commands': [],
                'joint_pos': [],
                'joint_vel': [],
                'actions': [],
                'max_history_length': 300,  # 增加历史长度以适应复杂船舶运动
            }
            
        # 初始化平台状态历史（用于训练目标）
        if not hasattr(self, '_platform_state_history'):
            self._platform_state_history = {
                'roll': [],
                'pitch': [],
                'roll_ang_vel': [],
                'pitch_ang_vel': [],
                'max_history_length': 300,  # 与观测历史一致
            }
        
        # 获取当前观测
        base_lin_vel = robot.data.root_lin_vel_b.clone()  # [num_envs, 3]
        base_ang_vel = robot.data.root_ang_vel_b.clone()  # [num_envs, 3]
        projected_gravity = robot.data.projected_gravity_b.clone()  # [num_envs, 3]
        joint_pos = robot.data.joint_pos.clone()  # [num_envs, num_joints]
        joint_vel = robot.data.joint_vel.clone()  # [num_envs, num_joints]
        
        # 获取速度命令
        try:
            velocity_commands = self.command_manager.get_command("base_velocity")  # [num_envs, 3] 或 [num_envs, ...]
            if velocity_commands.dim() == 1:
                velocity_commands = velocity_commands.unsqueeze(-1)  # [num_envs, 1]
        except:
            # 如果没有速度命令，使用零
            velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 获取当前动作（从上一步）
        if hasattr(self, '_last_actions') and self._last_actions is not None:
            actions = self._last_actions.clone()  # [num_envs, num_actions]
        else:
            num_actions = robot.num_joints
            actions = torch.zeros(self.num_envs, num_actions, device=self.device)
        
        # 添加到观测历史
        self._observation_history['base_lin_vel'].append(base_lin_vel)
        self._observation_history['base_ang_vel'].append(base_ang_vel)
        self._observation_history['projected_gravity'].append(projected_gravity)
        self._observation_history['velocity_commands'].append(velocity_commands)
        self._observation_history['joint_pos'].append(joint_pos)
        self._observation_history['joint_vel'].append(joint_vel)
        self._observation_history['actions'].append(actions)
        
        # 收集平台状态历史（用于训练目标）
        try:
            platform = self.scene["platform"]
            from isaaclab.utils.math import euler_xyz_from_quat
            current_quat = platform.data.root_quat_w
            current_ang_vel = platform.data.root_ang_vel_w
            current_roll, current_pitch, _ = euler_xyz_from_quat(current_quat)
            current_roll_ang_vel = current_ang_vel[:, 0]
            current_pitch_ang_vel = current_ang_vel[:, 1]
            
            # 只存储环境0的状态（所有环境相同）
            self._platform_state_history['roll'].append(current_roll[0:1])
            self._platform_state_history['pitch'].append(current_pitch[0:1])
            self._platform_state_history['roll_ang_vel'].append(current_roll_ang_vel[0:1])
            self._platform_state_history['pitch_ang_vel'].append(current_pitch_ang_vel[0:1])
            
            # 限制历史长度
            max_len = self._platform_state_history['max_history_length']
            for key in ['roll', 'pitch', 'roll_ang_vel', 'pitch_ang_vel']:
                if len(self._platform_state_history[key]) > max_len:
                    self._platform_state_history[key].pop(0)
        except KeyError:
            pass  # 如果没有平台，跳过
        
        # 限制历史长度
        max_len = self._observation_history['max_history_length']
        for key in self._observation_history.keys():
            if key != 'max_history_length' and len(self._observation_history[key]) > max_len:
                self._observation_history[key].pop(0)
    
    def _infer_platform_motion_from_observations(self):
        """从观测信息推断平台运动"""
        # 检查是否启用推断模式（可以通过配置控制）
        # 如果直接观测平台可用，则不需要推断
        use_inference = getattr(self, 'use_platform_inference', False)
        if not use_inference:
            return
        
        # 初始化推断器（如果还没有）
        if not hasattr(self, '_platform_inference'):
            self._init_platform_inference()
            return
        
        # 检查观测历史是否足够
        if len(self._observation_history.get('base_lin_vel', [])) < self._platform_inference.history_length:
            return
        
        # 准备观测历史（只取环境0，因为所有环境的平台运动相同）
        history_length = self._platform_inference.history_length
        
        # 确保所有历史数据都有相同的长度
        recent_base_lin_vel = self._observation_history['base_lin_vel'][-history_length:]
        recent_base_ang_vel = self._observation_history['base_ang_vel'][-history_length:]
        recent_projected_gravity = self._observation_history['projected_gravity'][-history_length:]
        recent_velocity_commands = self._observation_history['velocity_commands'][-history_length:]
        recent_joint_pos = self._observation_history['joint_pos'][-history_length:]
        recent_joint_vel = self._observation_history['joint_vel'][-history_length:]
        recent_actions = self._observation_history['actions'][-history_length:]
        recent_height_scan = self._observation_history['height_scan'][-history_length:]
        
        obs_history = {
            'base_lin_vel': torch.stack(recent_base_lin_vel, dim=1)[0:1],  # [1, history_length, 3]
            'base_ang_vel': torch.stack(recent_base_ang_vel, dim=1)[0:1],  # [1, history_length, 3]
            'projected_gravity': torch.stack(recent_projected_gravity, dim=1)[0:1],  # [1, history_length, 3]
            'velocity_commands': torch.stack(recent_velocity_commands, dim=1)[0:1],  # [1, history_length, num_velocity_commands]
            'joint_pos': torch.stack(recent_joint_pos, dim=1)[0:1],  # [1, history_length, num_joints]
            'joint_vel': torch.stack(recent_joint_vel, dim=1)[0:1],  # [1, history_length, num_joints]
            'actions': torch.stack(recent_actions, dim=1)[0:1],  # [1, history_length, num_actions]
            'height_scan': torch.stack(recent_height_scan, dim=1)[0:1] if len(recent_height_scan) > 0 and recent_height_scan[0].numel() > 0 else torch.zeros(1, history_length, 0, device=self.device),  # [1, history_length, num_height_scan_points]
        }
        
        # 推断平台运动（只推断环境0，然后广播到所有环境）
        inferred_platform = self._platform_inference.infer_platform_motion(obs_history)  # [1, 4]
        
        # 广播到所有环境
        num_envs = self.num_envs
        inferred_platform = inferred_platform.expand(num_envs, -1)  # [num_envs, 4]
        
        # 添加到推断的平台历史
        if not hasattr(self, '_inferred_platform_history'):
            self._inferred_platform_history = {
                'roll': [],
                'pitch': [],
                'roll_ang_vel': [],
                'pitch_ang_vel': [],
                'max_history_length': 150,
            }
        
        self._inferred_platform_history['roll'].append(inferred_platform[:, 0])
        self._inferred_platform_history['pitch'].append(inferred_platform[:, 1])
        self._inferred_platform_history['roll_ang_vel'].append(inferred_platform[:, 2])
        self._inferred_platform_history['pitch_ang_vel'].append(inferred_platform[:, 3])
        
        # 限制历史长度
        max_len = self._inferred_platform_history['max_history_length']
        for key in ['roll', 'pitch', 'roll_ang_vel', 'pitch_ang_vel']:
            if len(self._inferred_platform_history[key]) > max_len:
                self._inferred_platform_history[key].pop(0)
        
        # 如果训练时可以使用真实平台数据，则更新推断器
        try:
            platform = self.scene["platform"]
            from isaaclab.utils.math import euler_xyz_from_quat
            
            # 获取真实平台运动（用于训练）
            current_quat = platform.data.root_quat_w
            current_ang_vel = platform.data.root_ang_vel_w
            current_roll, current_pitch, _ = euler_xyz_from_quat(current_quat)
            current_roll_ang_vel = current_ang_vel[:, 0]
            current_pitch_ang_vel = current_ang_vel[:, 1]
            
            true_platform_motion = torch.stack([
                current_roll[0:1],  # 只取环境0
                current_pitch[0:1],
                current_roll_ang_vel[0:1],
                current_pitch_ang_vel[0:1],
            ], dim=1)  # [1, 4]
            
            # 更新推断器（在线学习）
            self._platform_inference.update(obs_history, true_platform_motion)
        except KeyError:
            pass  # 如果没有平台，跳过训练
    
    def _update_platform_predictor(self):
        """更新平台预测器：训练网络（使用机器狗观测历史预测未来平台运动）"""
        try:
            platform = self.scene["platform"]
            robot = self.scene["robot"]
        except KeyError:
            return
        
        # 初始化预测器（如果还没有）
        if not hasattr(self, '_platform_predictor'):
            self._init_platform_predictor()
            return
        
        # 检查观测历史是否足够
        if not hasattr(self, '_observation_history') or len(self._observation_history.get('base_lin_vel', [])) < self._platform_predictor.history_length:
            return
        
        # 获取当前平台状态（作为训练目标）
            from isaaclab.utils.math import euler_xyz_from_quat
            current_quat = platform.data.root_quat_w
            current_ang_vel = platform.data.root_ang_vel_w
            current_roll, current_pitch, _ = euler_xyz_from_quat(current_quat)
            current_roll_ang_vel = current_ang_vel[:, 0]
            current_pitch_ang_vel = current_ang_vel[:, 1]
            
        # 准备机器狗观测历史（只取环境0，因为所有环境的平台运动相同）
        history_length = self._platform_predictor.history_length
        recent_base_lin_vel = self._observation_history['base_lin_vel'][-history_length:]
        recent_base_ang_vel = self._observation_history['base_ang_vel'][-history_length:]
        recent_projected_gravity = self._observation_history['projected_gravity'][-history_length:]
        recent_velocity_commands = self._observation_history['velocity_commands'][-history_length:]
        recent_joint_pos = self._observation_history['joint_pos'][-history_length:]
        recent_joint_vel = self._observation_history['joint_vel'][-history_length:]
        recent_actions = self._observation_history['actions'][-history_length:]
        
        # 堆叠为tensor（只取环境0）
        obs_history = {
            'base_lin_vel': torch.stack(recent_base_lin_vel, dim=1)[0:1],  # [1, history_length, 3]
            'base_ang_vel': torch.stack(recent_base_ang_vel, dim=1)[0:1],  # [1, history_length, 3]
            'projected_gravity': torch.stack(recent_projected_gravity, dim=1)[0:1],  # [1, history_length, 3]
            'velocity_commands': torch.stack(recent_velocity_commands, dim=1)[0:1],  # [1, history_length, num_velocity_commands]
            'joint_pos': torch.stack(recent_joint_pos, dim=1)[0:1],  # [1, history_length, num_joints]
            'joint_vel': torch.stack(recent_joint_vel, dim=1)[0:1],  # [1, history_length, num_joints]
            'actions': torch.stack(recent_actions, dim=1)[0:1],  # [1, history_length, num_actions]
        }
        
        # ========== 训练逻辑：使用机器狗观测信息预测平台未来运动 ==========
        # 核心思想：
        # - 输入：历史中某个时间点t的机器狗观测信息（关节位置、关节速度、基座速度、动作等）
        # - 目标：时间点t之后（未来）的平台状态（roll, pitch, roll_ang_vel, pitch_ang_vel）
        # - 这样网络学习：根据机器狗的运动信息，预测平台未来的运动
        
        prediction_steps = self._platform_predictor.prediction_steps
        
        # 检查是否有足够的历史数据（至少history_length + prediction_steps步）
        min_required_history = history_length + prediction_steps
        if not hasattr(self, '_platform_state_history') or len(self._platform_state_history['roll']) < min_required_history:
            return  # 历史数据不够，跳过训练
        
        # 获取平台状态历史
        roll_history = self._platform_state_history['roll']
        pitch_history = self._platform_state_history['pitch']
        roll_ang_vel_history = self._platform_state_history['roll_ang_vel']
        pitch_ang_vel_history = self._platform_state_history['pitch_ang_vel']
        
        # 获取机器狗观测历史（需要足够长的历史）
        if len(self._observation_history['base_lin_vel']) < min_required_history:
            return
        
        # ========== 改进的训练数据采样策略 ==========
        # 目标：从历史数据中选择多样化的训练样本，确保网络学习到不同运动模式
        
        # 对于历史中的每个时间点t（从history_length到len-prediction_steps），
        # 使用t时刻及之前的机器狗观测作为输入，t时刻之后的平台状态作为目标
        
        start_idx = history_length  # 最早可以使用的训练时间点
        end_idx = len(roll_history) - prediction_steps  # 最晚可以使用的训练时间点
        if end_idx <= start_idx:
            return
        
        # 改进1：大幅增加训练样本数量，提高数据利用率和训练效率
        # 关键改进：从每次10-20个样本增加到50-100个样本，加快学习速度
        available_samples = end_idx - start_idx + 1
        num_training_samples = min(100, max(20, available_samples // 5))  # 每次训练20-100个样本（从10-20增加到20-100）
        
        if num_training_samples <= 0:
            return
        
        # 改进2：使用分层采样策略，确保覆盖不同时间段
        # - 30%来自最新数据（最近25%的历史）
        # - 40%来自中间数据（中间50%的历史）
        # - 30%来自较旧数据（最早25%的历史）
        recent_start = max(start_idx, int(end_idx - (end_idx - start_idx) * 0.25))
        middle_start = max(start_idx, int(start_idx + (end_idx - start_idx) * 0.25))
        middle_end = min(end_idx, int(start_idx + (end_idx - start_idx) * 0.75))
        old_end = min(end_idx, int(start_idx + (end_idx - start_idx) * 0.25))
        
        training_indices = []
        
        # 最新数据（30%）
        num_recent = max(1, int(num_training_samples * 0.3))
        if recent_start <= end_idx:
            recent_indices = torch.linspace(recent_start, end_idx, num_recent + 1, dtype=torch.long)[:-1].tolist()
            training_indices.extend(recent_indices)
        
        # 中间数据（40%）
        num_middle = max(1, int(num_training_samples * 0.4))
        if middle_start < middle_end:
            middle_indices = torch.linspace(middle_start, middle_end, num_middle + 1, dtype=torch.long)[:-1].tolist()
            training_indices.extend(middle_indices)
        
        # 较旧数据（30%）
        num_old = max(1, num_training_samples - len(training_indices))
        if start_idx < old_end:
            old_indices = torch.linspace(start_idx, old_end, num_old + 1, dtype=torch.long)[:-1].tolist()
            training_indices.extend(old_indices)
        
        # 去重并排序
        training_indices = sorted(list(set(training_indices)))
        training_indices = [idx for idx in training_indices if start_idx <= idx <= end_idx]
        
        if len(training_indices) == 0:
            return
        
        # 为每个训练样本准备数据
        for t_idx in training_indices:
            # 输入：t_idx时刻之前的机器狗观测历史（包括t_idx时刻）
            # 使用t_idx-history_length+1到t_idx的数据（共history_length个时间步）
            input_start_idx = max(0, t_idx - history_length + 1)
            input_end_idx = t_idx + 1  # 包括t_idx时刻
            
            # 提取机器狗观测历史（只使用环境0）
            obs_input = {
                'base_lin_vel': torch.stack(self._observation_history['base_lin_vel'][input_start_idx:input_end_idx], dim=1)[0:1],  # [1, actual_length, 3]
                'base_ang_vel': torch.stack(self._observation_history['base_ang_vel'][input_start_idx:input_end_idx], dim=1)[0:1],
                'projected_gravity': torch.stack(self._observation_history['projected_gravity'][input_start_idx:input_end_idx], dim=1)[0:1],
                'velocity_commands': torch.stack(self._observation_history['velocity_commands'][input_start_idx:input_end_idx], dim=1)[0:1],
                'joint_pos': torch.stack(self._observation_history['joint_pos'][input_start_idx:input_end_idx], dim=1)[0:1],
                'joint_vel': torch.stack(self._observation_history['joint_vel'][input_start_idx:input_end_idx], dim=1)[0:1],
                'actions': torch.stack(self._observation_history['actions'][input_start_idx:input_end_idx], dim=1)[0:1],
            }
            
            # 如果历史长度不足，需要填充（在时间维度前填充）
            actual_length = input_end_idx - input_start_idx
            if actual_length < history_length:
                padding_size = history_length - actual_length
                for key in obs_input.keys():
                    # 在时间维度前填充（重复第一个时间步）
                    first_step = obs_input[key][:, 0:1, :]  # [1, 1, ...]
                    padding = first_step.expand(1, padding_size, -1)  # [1, padding_size, ...]
                    obs_input[key] = torch.cat([padding, obs_input[key]], dim=1)  # [1, history_length, ...]
            
            # 目标：t_idx时刻之后（未来）的平台状态（prediction_steps步）
            # 从观测输入中获取设备信息
            device = obs_input['base_lin_vel'].device
            # 2维：roll, pitch（只预测这两个角度）
            future_states = torch.zeros(1, prediction_steps, 2, device=device)
            
            for step in range(prediction_steps):
                target_idx = t_idx + step + 1  # t_idx+1是未来第1步，t_idx+2是未来第2步，...
                if target_idx < len(roll_history):
                    # 使用真实的未来平台状态（从历史数据中获取）
                    future_states[0, step, 0] = roll_history[target_idx][0]  # roll
                    future_states[0, step, 1] = pitch_history[target_idx][0]  # pitch
                else:
                    # 如果超出历史范围，使用最后一个状态
                    future_states[0, step, 0] = roll_history[-1][0]
                    future_states[0, step, 1] = pitch_history[-1][0]
            
            # 更新网络：使用机器狗观测信息预测平台未来运动
            self._platform_predictor.update_from_observations(
                obs_input,  # [1, history_length, ...] - 机器狗的观测信息（关节位置、速度、基座速度、动作等）
                future_states  # [1, prediction_steps, 2] - 平台未来的状态（roll, pitch）
            )
        
        # ========== 定期评估预测质量 ==========
        # 如果预测质量还未验证通过，则进行评估
        if not self._platform_predictor.prediction_quality_verified:
            # 关键优化：大幅减少评估频率，提高RL训练速度
            # 每隔2000步评估一次（从500增加到2000，减少计算开销）
            if not hasattr(self, '_last_prediction_evaluation_step'):
                self._last_prediction_evaluation_step = -1
            
            current_step = getattr(self, '_sim_step_counter', 0)
            if current_step - self._last_prediction_evaluation_step >= 2000:
                self._last_prediction_evaluation_step = current_step
                
                # 检查历史数据是否足够（需要足够长的平台历史数据用于评估）
                # 注意：评估需要使用平台历史数据，而不是机器狗观测历史
                platform_history = self.get_platform_history(history_length=None)
                
                if platform_history.get('quat_w') is not None and platform_history.get('ang_vel_w') is not None:
                    # 检查历史数据长度
                    quat_history = platform_history['quat_w']
                    if isinstance(quat_history, list):
                        total_history_length = len(quat_history)
                    else:
                        total_history_length = quat_history.shape[0]
                    
                    # 需要至少 history_length + min_evaluation_samples 个历史点
                    min_required_length = self._platform_predictor.history_length + self._platform_predictor.min_evaluation_samples
                    
                    if total_history_length >= min_required_length:
                        # 评估预测质量（使用机器狗观测历史和平台历史数据）
                        # 检查观测历史是否足够
                        if hasattr(self, '_observation_history') and len(self._observation_history.get('base_lin_vel', [])) >= self._platform_predictor.history_length:
                            is_good = self._platform_predictor.evaluate_prediction_quality_from_observations(
                                observation_history=self._observation_history,
                                platform_history=platform_history,
                                delay_steps=5  # 与训练时保持一致
                            )
                        else:
                            # 观测历史不足，无法评估
                            if current_step % 2000 == 0:
                                print(f"[预测质量评估] 观测历史数据不足，无法进行评估")
                            is_good = False
                        
                        # 如果评估通过，设置标志，之后就一直使用网络预测
                        if is_good:
                            self._platform_predictor.prediction_quality_verified = True
                    else:
                        # 历史数据不足，不进行评估
                        if current_step % 2000 == 0:  # 每2000步打印一次，避免打印太频繁
                            print(f"[预测质量评估] 历史数据不足: {total_history_length} < {min_required_length}，无法进行评估")
                else:
                    # 平台历史数据不可用
                    if current_step % 2000 == 0:  # 每2000步打印一次
                        print(f"[预测质量评估] 平台历史数据不可用，无法进行评估")
        
        # ========== 关键改进：只有当使用神经网络预测时才评估和替换生产模型 ==========
        # 如果使用线性外推（prediction_quality_good=False），让神经网络自由更新，不评估和替换
        # 只有当使用神经网络预测（prediction_quality_good=True）时，才需要评估候选模型，
        # 只有更好的参数才替换生产模型，确保奖励函数使用的是最好的模型
        if self._platform_predictor.prediction_quality_verified:
            # 关键优化：减少生产模型评估频率，提高RL训练速度
            # 每隔一定步数评估一次候选模型，只有评估通过才替换生产模型
            current_step = getattr(self, '_sim_step_counter', 0)
            
            # 关键优化：不是每步都评估，而是每隔一定步数才评估
            if not hasattr(self, '_last_production_model_evaluation_step'):
                self._last_production_model_evaluation_step = -1
            
            # 每隔2000步评估一次（与预测质量评估保持一致，减少计算开销）
            if current_step - self._last_production_model_evaluation_step >= 2000:
                self._last_production_model_evaluation_step = current_step
                
                # 获取平台历史数据用于评估
                platform_history = self.get_platform_history(history_length=None)
                
                if platform_history.get('quat_w') is not None and platform_history.get('ang_vel_w') is not None:
                    # 关键修复：传递观测历史，因为模型现在需要观测历史作为输入
                    observation_history = getattr(self, '_observation_history', None)
                    if observation_history is not None:
                        self._platform_predictor.evaluate_and_update_production_model(
                            platform_history,
                            delay_steps=5,
                            current_step=current_step,
                            observation_history=observation_history
                        )
    
    def get_platform_prediction_from_observations(self, prediction_steps: int = 1) -> dict[str, torch.Tensor] | None:
        """从机器狗观测历史获取平台运动预测结果（关键功能）
        
        使用神经网络从机器狗的观测历史（基座速度、角速度、重力投影、关节状态等）预测平台运动。
        这是关键功能：让机器狗从自身观测学习预测平台运动规律。
        
        Args:
            prediction_steps: 预测未来多少步（默认1步，即下一步）
        
        Returns:
            预测结果字典，包含roll和pitch：
            - 'roll': 预测的roll角度 [num_envs]
            - 'pitch': 预测的pitch角度 [num_envs]
            - 'roll_ang_vel': 预测的roll角速度 [num_envs] (暂时为0，可后续添加)
            - 'pitch_ang_vel': 预测的pitch角速度 [num_envs] (暂时为0，可后续添加)
            如果预测器未初始化或观测历史不足，返回None
            
        注意：只预测roll和pitch，因为跟随任务只需要保持XY平面平行。
        x, y, z, yaw对跟随任务没有帮助，反而可能引入噪声，增加学习难度。
        """
        if not hasattr(self, '_platform_predictor'):
            return None
        
        # 检查观测历史是否足够
        if not hasattr(self, '_observation_history') or len(self._observation_history.get('base_lin_vel', [])) < self._platform_predictor.history_length:
            return None
        
        # 准备机器狗观测历史（只取环境0，因为所有环境的平台运动相同）
        history_length = self._platform_predictor.history_length
        recent_base_lin_vel = self._observation_history['base_lin_vel'][-history_length:]
        recent_base_ang_vel = self._observation_history['base_ang_vel'][-history_length:]
        recent_projected_gravity = self._observation_history['projected_gravity'][-history_length:]
        recent_velocity_commands = self._observation_history['velocity_commands'][-history_length:]
        recent_joint_pos = self._observation_history['joint_pos'][-history_length:]
        recent_joint_vel = self._observation_history['joint_vel'][-history_length:]
        recent_actions = self._observation_history['actions'][-history_length:]
        
        # 堆叠为tensor（只取环境0）
        obs_history = {
            'base_lin_vel': torch.stack(recent_base_lin_vel, dim=1)[0:1],  # [1, history_length, 3]
            'base_ang_vel': torch.stack(recent_base_ang_vel, dim=1)[0:1],  # [1, history_length, 3]
            'projected_gravity': torch.stack(recent_projected_gravity, dim=1)[0:1],  # [1, history_length, 3]
            'velocity_commands': torch.stack(recent_velocity_commands, dim=1)[0:1],  # [1, history_length, num_velocity_commands]
            'joint_pos': torch.stack(recent_joint_pos, dim=1)[0:1],  # [1, history_length, num_joints]
            'joint_vel': torch.stack(recent_joint_vel, dim=1)[0:1],  # [1, history_length, num_joints]
            'actions': torch.stack(recent_actions, dim=1)[0:1],  # [1, history_length, num_actions]
        }
        
        # 使用神经网络预测（从机器狗观测历史预测平台运动）
        try:
            predicted_roll, predicted_pitch, predicted_roll_ang_vel, predicted_pitch_ang_vel = \
                self._platform_predictor.predict_future_from_observations(
                    obs_history,
                    prediction_steps=prediction_steps
                )
            
            # 广播到所有环境
            num_envs = self.num_envs
            predicted_roll = predicted_roll.expand(num_envs)
            predicted_pitch = predicted_pitch.expand(num_envs)
            predicted_roll_ang_vel = predicted_roll_ang_vel.expand(num_envs)
            predicted_pitch_ang_vel = predicted_pitch_ang_vel.expand(num_envs)
            
            return {
                'roll': predicted_roll,
                'pitch': predicted_pitch,
                'roll_ang_vel': predicted_roll_ang_vel,
                'pitch_ang_vel': predicted_pitch_ang_vel
            }
        except Exception as e:
            # 如果预测失败，返回None
            if hasattr(self, '_last_prediction_error_step'):
                current_step = getattr(self, '_sim_step_counter', 0)
                if current_step - self._last_prediction_error_step >= 1000:
                    print(f"[平台预测错误] 从观测预测平台运动失败: {e}")
                    self._last_prediction_error_step = current_step
            else:
                self._last_prediction_error_step = getattr(self, '_sim_step_counter', 0)
            return None
    
    def get_platform_prediction(self) -> dict[str, torch.Tensor] | None:
        """获取平台运动预测结果（使用平台历史数据预测，已废弃，保留仅为兼容性）
        
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
    
    def get_platform_future_prediction(self, prediction_time: float = 0.1) -> dict[str, torch.Tensor] | None:
        """获取未来时刻的平台预测结果（使用机器狗观测历史预测未来状态）
        
        这个方法专门用于奖励函数：使用机器狗观测历史来预测未来时刻的平台状态。
        
        Args:
            prediction_time: 预测时间范围（秒），例如0.1秒表示预测未来0.1秒后的状态
        
        Returns:
            预测结果字典，包含：
            - 'roll': 预测的未来roll角度 [num_envs]
            - 'pitch': 预测的未来pitch角度 [num_envs]
            - 'roll_ang_vel': 预测的未来roll角速度 [num_envs]
            - 'pitch_ang_vel': 预测的未来pitch角速度 [num_envs]
            如果预测器未初始化或历史数据不足，返回None
        """
        if not hasattr(self, '_platform_predictor'):
            return None
        
        # 检查观测历史是否足够
        if not hasattr(self, '_observation_history') or len(self._observation_history.get('base_lin_vel', [])) < self._platform_predictor.history_length:
            return None
        
        # 准备机器狗观测历史（只取环境0，因为所有环境的平台运动相同）
        history_length = self._platform_predictor.history_length
        recent_base_lin_vel = self._observation_history['base_lin_vel'][-history_length:]
        recent_base_ang_vel = self._observation_history['base_ang_vel'][-history_length:]
        recent_projected_gravity = self._observation_history['projected_gravity'][-history_length:]
        recent_velocity_commands = self._observation_history['velocity_commands'][-history_length:]
        recent_joint_pos = self._observation_history['joint_pos'][-history_length:]
        recent_joint_vel = self._observation_history['joint_vel'][-history_length:]
        recent_actions = self._observation_history['actions'][-history_length:]
        
        # 堆叠为tensor（只取环境0）
        obs_history = {
            'base_lin_vel': torch.stack(recent_base_lin_vel, dim=1)[0:1],  # [1, history_length, 3]
            'base_ang_vel': torch.stack(recent_base_ang_vel, dim=1)[0:1],  # [1, history_length, 3]
            'projected_gravity': torch.stack(recent_projected_gravity, dim=1)[0:1],  # [1, history_length, 3]
            'velocity_commands': torch.stack(recent_velocity_commands, dim=1)[0:1],  # [1, history_length, num_velocity_commands]
            'joint_pos': torch.stack(recent_joint_pos, dim=1)[0:1],  # [1, history_length, num_joints]
            'joint_vel': torch.stack(recent_joint_vel, dim=1)[0:1],  # [1, history_length, num_joints]
            'actions': torch.stack(recent_actions, dim=1)[0:1],  # [1, history_length, num_actions]
        }
        
        # 使用机器狗观测历史预测未来状态
        dt = self.step_dt if hasattr(self, 'step_dt') else 0.02
        prediction_steps = int(prediction_time / dt)
        predicted_roll, predicted_pitch, predicted_roll_ang_vel, predicted_pitch_ang_vel = \
            self._platform_predictor.predict_future_from_observations(
                obs_history, prediction_steps=prediction_steps
            )
        
        # 调试信息：定期打印预测值，确认预测是否在变化
        if not hasattr(self, '_last_prediction_debug_step'):
            self._last_prediction_debug_step = -1
            self._last_prediction_values = None
        
        current_step = getattr(self, '_sim_step_counter', 0)
        if current_step - self._last_prediction_debug_step >= 500:  # 每500步打印一次
            # 计算预测值的统计信息
            pred_roll_mean = predicted_roll.mean().item()
            pred_roll_std = predicted_roll.std().item()
            pred_pitch_mean = predicted_pitch.mean().item()
            pred_pitch_std = predicted_pitch.std().item()
            
            # 检查预测值是否在变化
            if self._last_prediction_values is not None:
                roll_diff = abs(pred_roll_mean - self._last_prediction_values[0])
                pitch_diff = abs(pred_pitch_mean - self._last_prediction_values[1])
                is_changing = (roll_diff > 1e-4) or (pitch_diff > 1e-4)
            else:
                is_changing = True
            
            print(f"[神经网络预测调试] 步骤 {current_step}: "
                  f"预测roll均值={pred_roll_mean:.6f}±{pred_roll_std:.6f}, "
                  f"预测pitch均值={pred_pitch_mean:.6f}±{pred_pitch_std:.6f}, "
                  f"预测值是否变化={is_changing}, "
                  f"训练步数={getattr(self._platform_predictor, 'candidate_train_steps', 0)}")
            
            self._last_prediction_debug_step = current_step
            self._last_prediction_values = (pred_roll_mean, pred_pitch_mean)
        
        return {
            'roll': predicted_roll,
            'pitch': predicted_pitch,
            'roll_ang_vel': predicted_roll_ang_vel,
            'pitch_ang_vel': predicted_pitch_ang_vel
        }
    
    def is_platform_prediction_quality_good(self) -> bool:
        """检查平台预测器的预测质量是否足够好
        
        只有当预测器收集了足够多的样本且平均误差小于阈值时，才认为预测质量足够好。
        这样可以避免在预测器还未训练好时使用不准确的预测。
        
        Returns:
            True: 预测质量足够好，可以使用神经网络预测
            False: 预测质量不够好，应该使用线性外推
        """
        if not hasattr(self, '_platform_predictor'):
            return False
        return self._platform_predictor.is_prediction_quality_good()
    
    def get_platform_prediction_quality_info(self) -> dict:
        """获取平台预测质量的详细信息
        
        Returns:
            包含预测质量信息的字典，如果预测器未初始化则返回None
        """
        if not hasattr(self, '_platform_predictor'):
            return None
        return self._platform_predictor.get_prediction_quality_info()
    
    def extrapolate_platform_future_advanced(
        self,
        prediction_time: float = 0.1,
        history_window: int = 50,  # 大幅增加历史窗口：从20增加到50，使用更多历史数据来处理复杂运动
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """通用的自适应外推方法预测未来平台状态（基于历史姿态和角速度数据）
        
        通用方法（不假设运动类型）：
        1. 自适应多项式外推：根据历史数据质量自动选择最优阶数（1-4阶）
        2. 加权历史数据：使用指数衰减权重，更重视最近的数据
        3. 多方法融合：结合角度外推和角速度积分，提高鲁棒性
        4. 数值稳定性：自动检测和避免数值不稳定
        
        Args:
            prediction_time: 预测时间范围（秒）
            history_window: 用于估计的历史窗口大小（时间步数）
        
        Returns:
            future_roll: 预测的未来roll角度 [num_envs]
            future_pitch: 预测的未来pitch角度 [num_envs]
        """
        from isaaclab.utils.math import euler_xyz_from_quat
        
        try:
            platform = self.scene["platform"]
        except KeyError:
            num_envs = 1
            device = self.device
            return (
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device)
            )
        
        # 获取当前状态
        current_quat = platform.data.root_quat_w
        current_ang_vel = platform.data.root_ang_vel_w
        
        current_roll, current_pitch, _ = euler_xyz_from_quat(current_quat)
        current_roll_ang_vel = current_ang_vel[:, 0]
        current_pitch_ang_vel = current_ang_vel[:, 1]
        
        # 获取历史数据
        platform_history = self.get_platform_history(history_length=history_window + 1)
        
        if platform_history.get('ang_vel_w') is None or len(platform_history['ang_vel_w']) < 2:
            # 如果历史数据不足，使用简单的一阶外推
            future_roll = current_roll + current_roll_ang_vel * prediction_time
            future_pitch = current_pitch + current_pitch_ang_vel * prediction_time
            return future_roll, future_pitch
        
        # 处理历史数据
        ang_vel_history = platform_history['ang_vel_w']
        quat_history = platform_history.get('quat_w')
        
        if isinstance(ang_vel_history, list):
            ang_vel_history = torch.stack(ang_vel_history, dim=0)  # [history_length, num_envs, 3]
        if quat_history is not None and isinstance(quat_history, list):
            quat_history = torch.stack(quat_history, dim=0)  # [history_length, num_envs, 4]
        
        # 提取最近的历史数据
        recent_history_length = min(history_window + 1, ang_vel_history.shape[0])
        recent_ang_vel = ang_vel_history[-recent_history_length:]  # [recent_history_length, num_envs, 3]
        
        num_envs = recent_ang_vel.shape[1]
        dt = self.step_dt if hasattr(self, 'step_dt') else 0.02
        
        # 关键改进：所有平台的运动都是完全一致的，只需要计算一次（环境0），然后广播到所有环境
        # 提取roll和pitch角速度历史（只使用环境0）
        roll_ang_vel_history = recent_ang_vel[:, 0, 0]  # [recent_history_length] 只使用环境0
        pitch_ang_vel_history = recent_ang_vel[:, 0, 1]  # [recent_history_length] 只使用环境0
        
        # ========== 改进的加权中心差分估计角加速度 ==========
        # 使用更多历史点和更长的窗口来提高估计精度
        # 只计算一次（环境0），然后广播到所有环境
        roll_ang_acc_single = torch.tensor(0.0, device=current_roll.device)
        pitch_ang_acc_single = torch.tensor(0.0, device=current_pitch.device)
        
        # 最优中心差分阶数选择（根据运动类型自适应）
        # 
        # 【简单运动（如正弦运动）】：3阶最优
        #   - 1阶：误差O(h²)，需要3个点，精度低
        #   - 2阶：误差O(h⁴)，需要5个点，精度中等
        #   - 3阶：误差O(h⁶)，需要7个点，精度高，对噪声不敏感（最佳平衡点）
        #   - 4阶及以上：精度提升有限，但噪声敏感性显著增加
        #
        # 【复杂动力学运动（如船舶运动）】：2-3阶最优
        #   船舶运动特点：
        #   - 6自由度耦合（surge, sway, heave, roll, pitch, yaw）
        #   - 非线性动力学（非线性阻尼、记忆效应）
        #   - 波浪干扰（多频率成分、随机性）
        #   - 多时间尺度（快速响应+慢速漂移）
        #   - 数值噪声和不确定性更大
        #
        #   选择理由：
        #   - 2阶：误差O(h⁴)，需要5个点，对噪声不敏感，适合有波浪干扰的情况
        #   - 3阶：误差O(h⁶)，需要7个点，精度更高，适合相对平滑的船舶运动
        #   - 4阶及以上：虽然精度更高，但对波浪干扰和数值噪声过于敏感，不推荐
        #
        #   推荐：对于船舶运动，使用2-3阶中心差分（根据历史数据长度自适应）
        #   - 历史数据充足（≥7点）：使用3阶，捕捉更复杂的加速度变化
        #   - 历史数据较少（5-6点）：使用2阶，更稳定
        #   - 历史数据很少（<5点）：使用1阶或前向差分
        #
        # 自适应选择：根据历史数据长度和运动复杂度选择
        if recent_history_length >= 7:
            # 历史数据充足，使用3阶（适合复杂动力学，如船舶运动）
            max_center_diff_order = 3
        elif recent_history_length >= 5:
            # 历史数据中等，使用2阶（平衡精度和稳定性）
            max_center_diff_order = 2
        else:
            # 历史数据较少，使用1阶
            max_center_diff_order = 1
        
        # 确保不超过可用数据限制
        max_center_diff_order = min(max_center_diff_order, recent_history_length // 2)
        
        if recent_history_length >= 3:
            total_weight = 0.0
            
            # 使用更多阶的中心差分，提高估计精度
            for i in range(1, max_center_diff_order + 1):
                if recent_history_length > 2 * i:
                    # 中心差分：α ≈ (ω(t+i*dt) - ω(t-i*dt)) / (2*i*dt)
                    # 权重：使用指数衰减，越近的点权重越大，但也要考虑阶数的影响
                    # w = exp(-i/2) / (i + 1)，既考虑距离又考虑阶数
                    weight = torch.exp(torch.tensor(-i / 2.0, device=current_roll.device)) / (i + 1)
                    
                    # Roll方向（只使用环境0）
                    roll_diff = roll_ang_vel_history[-1-i] - roll_ang_vel_history[-1+i]
                    roll_ang_acc_single += weight * roll_diff / (2 * i * dt)
                    
                    # Pitch方向（只使用环境0）
                    pitch_diff = pitch_ang_vel_history[-1-i] - pitch_ang_vel_history[-1+i]
                    pitch_ang_acc_single += weight * pitch_diff / (2 * i * dt)
                    
                    total_weight += weight.item()
            
            # 归一化权重
            if total_weight > 1e-8:
                roll_ang_acc_single = roll_ang_acc_single / total_weight
                pitch_ang_acc_single = pitch_ang_acc_single / total_weight
            else:
                # 如果无法使用中心差分，使用简单的前向差分
                if recent_history_length >= 2:
                    roll_ang_acc_single = (roll_ang_vel_history[-1] - roll_ang_vel_history[-2]) / dt
                    pitch_ang_acc_single = (pitch_ang_vel_history[-1] - pitch_ang_vel_history[-2]) / dt
        elif recent_history_length >= 2:
            # 如果只有2个点，使用简单的前向差分
            roll_ang_acc_single = (roll_ang_vel_history[-1] - roll_ang_vel_history[-2]) / dt
            pitch_ang_acc_single = (pitch_ang_vel_history[-1] - pitch_ang_vel_history[-2]) / dt
        
        # 广播到所有环境（因为所有平台的运动都是完全一致的）
        roll_ang_acc = roll_ang_acc_single.expand(num_envs)
        pitch_ang_acc = pitch_ang_acc_single.expand(num_envs)
        
        # ========== 方法2：使用历史姿态数据的加权差分（改进：使用更多历史点） ==========
        # 从历史姿态计算角速度，然后使用加权差分估计角加速度
        # 关键改进：所有平台的运动都是完全一致的，只需要计算一次（环境0），然后广播到所有环境
        if quat_history is not None and recent_history_length >= 3:
            recent_quat = quat_history[-recent_history_length:, 0:1, :]  # 只使用环境0 [recent_history_length, 1, 4]
            
            # 提取历史roll和pitch角度（只提取环境0）
            history_roll = torch.zeros(recent_history_length, device=current_roll.device)  # [recent_history_length]
            history_pitch = torch.zeros(recent_history_length, device=current_pitch.device)  # [recent_history_length]
            
            for i in range(recent_history_length):
                roll_i, pitch_i, _ = euler_xyz_from_quat(recent_quat[i])
                history_roll[i] = roll_i[0]  # 只提取环境0
                history_pitch[i] = pitch_i[0]  # 只提取环境0
            
            # 从历史姿态计算角速度（前向差分）
            computed_roll_ang_vel = (history_roll[1:] - history_roll[:-1]) / dt  # [recent_history_length-1]
            computed_pitch_ang_vel = (history_pitch[1:] - history_pitch[:-1]) / dt  # [recent_history_length-1]
            
            # 使用加权中心差分从姿态计算的角速度估计角加速度（改进：使用更多阶）
            if computed_roll_ang_vel.shape[0] >= 3:
                roll_ang_acc_from_pose_single = torch.tensor(0.0, device=current_roll.device)
                pitch_ang_acc_from_pose_single = torch.tensor(0.0, device=current_pitch.device)
                total_weight_pose = 0.0
                
                # 自适应选择中心差分阶数（与角速度历史估计保持一致）
                # 对于复杂动力学运动（如船舶），根据可用数据自适应选择2-3阶
                if computed_roll_ang_vel.shape[0] >= 7:
                    max_order_pose = 3  # 历史数据充足，使用3阶
                elif computed_roll_ang_vel.shape[0] >= 5:
                    max_order_pose = 2  # 历史数据中等，使用2阶
                else:
                    max_order_pose = 1  # 历史数据较少，使用1阶
                
                max_order_pose = min(max_order_pose, computed_roll_ang_vel.shape[0] // 2)
                for i in range(1, max_order_pose + 1):
                    if computed_roll_ang_vel.shape[0] > 2 * i:
                        # 使用指数衰减权重
                        weight = torch.exp(torch.tensor(-i / 2.0, device=current_roll.device)) / (i + 1)
                        roll_diff = computed_roll_ang_vel[-1-i] - computed_roll_ang_vel[-1+i]
                        pitch_diff = computed_pitch_ang_vel[-1-i] - computed_pitch_ang_vel[-1+i]
                        roll_ang_acc_from_pose_single += weight * roll_diff / (2 * i * dt)
                        pitch_ang_acc_from_pose_single += weight * pitch_diff / (2 * i * dt)
                        total_weight_pose += weight.item()
                
                if total_weight_pose > 1e-8:
                    roll_ang_acc_from_pose_single = roll_ang_acc_from_pose_single / total_weight_pose
                    pitch_ang_acc_from_pose_single = pitch_ang_acc_from_pose_single / total_weight_pose
                    
                    # 广播到所有环境
                    roll_ang_acc_from_pose = roll_ang_acc_from_pose_single.expand(num_envs)
                    pitch_ang_acc_from_pose = pitch_ang_acc_from_pose_single.expand(num_envs)
                    
                    # 加权融合：70%来自角速度历史，30%来自姿态历史（提高角速度历史的权重）
                    roll_ang_acc = 0.7 * roll_ang_acc + 0.3 * roll_ang_acc_from_pose
                    pitch_ang_acc = 0.7 * pitch_ang_acc + 0.3 * pitch_ang_acc_from_pose
        
        # ========== 通用自适应多项式外推（不假设运动类型） ==========
        # 使用Taylor级数展开：θ(t+dt) = θ(t) + θ'(t)*dt + (1/2!)*θ''(t)*dt² + (1/3!)*θ'''(t)*dt³ + ...
        # 根据历史数据质量自适应选择最优阶数
        
        # 方法1：基于角速度积分的外推（改进版：考虑角加速度）
        # θ(t+dt) = θ(t) + ∫[t to t+dt] ω(τ) dτ
        # 使用角加速度估计未来角速度，然后积分
        if recent_history_length >= 3:
            # 估计未来角速度：ω(t+dt) = ω(t) + α(t)*dt
            # 使用已估计的角加速度
            roll_ang_vel_future = current_roll_ang_vel + roll_ang_acc * prediction_time
            pitch_ang_vel_future = current_pitch_ang_vel + pitch_ang_acc * prediction_time
            
            # 使用梯形积分（更准确）：θ(t+dt) ≈ θ(t) + 0.5*(ω(t) + ω(t+dt))*dt
            roll_from_vel_integration = current_roll + 0.5 * (current_roll_ang_vel + roll_ang_vel_future) * prediction_time
            pitch_from_vel_integration = current_pitch + 0.5 * (current_pitch_ang_vel + pitch_ang_vel_future) * prediction_time
        else:
            # 历史数据不足，使用简单线性外推
            roll_from_vel_integration = current_roll + current_roll_ang_vel * prediction_time
            pitch_from_vel_integration = current_pitch + current_pitch_ang_vel * prediction_time
        
        # 方法2：自适应阶数的Taylor级数外推
        # 根据历史数据质量自动选择最优阶数（1-4阶）
        max_order = min(4, recent_history_length // 2)  # 最多4阶，但受历史数据限制
        
        # 初始化外推结果
        roll_taylor = current_roll + current_roll_ang_vel * prediction_time  # 1阶（线性）
        pitch_taylor = current_pitch + current_pitch_ang_vel * prediction_time
        
        # 2阶：加入角加速度项
        if max_order >= 2 and recent_history_length >= 3:
            roll_taylor = roll_taylor + 0.5 * roll_ang_acc * (prediction_time ** 2)
            pitch_taylor = pitch_taylor + 0.5 * pitch_ang_acc * (prediction_time ** 2)
        
        # 3阶：加入角加加速度（jerk）项
        if max_order >= 3 and recent_history_length >= 6:
            # 估计角加加速度
            roll_jerk = torch.zeros(num_envs, device=current_roll.device)
            pitch_jerk = torch.zeros(num_envs, device=current_pitch.device)
            
            if recent_history_length >= 8:
                # 使用中心差分估计角加速度，然后计算jerk
                roll_ang_vel_recent = roll_ang_vel_history[-3:]
                pitch_ang_vel_recent = pitch_ang_vel_history[-3:]
                roll_acc_recent = (roll_ang_vel_recent[2] - roll_ang_vel_recent[0]) / (2 * dt)
                pitch_acc_recent = (pitch_ang_vel_recent[2] - pitch_ang_vel_recent[0]) / (2 * dt)
                
                roll_ang_vel_earlier = roll_ang_vel_history[-5:-2]
                pitch_ang_vel_earlier = pitch_ang_vel_history[-5:-2]
                roll_acc_earlier = (roll_ang_vel_earlier[2] - roll_ang_vel_earlier[0]) / (2 * dt)
                pitch_acc_earlier = (pitch_ang_vel_earlier[2] - pitch_ang_vel_earlier[0]) / (2 * dt)
                
                roll_jerk = (roll_acc_recent - roll_acc_earlier) / (2 * dt)
                pitch_jerk = (pitch_acc_recent - pitch_acc_earlier) / (2 * dt)
            
            roll_taylor = roll_taylor + (1.0 / 6.0) * roll_jerk * (prediction_time ** 3)
            pitch_taylor = pitch_taylor + (1.0 / 6.0) * pitch_jerk * (prediction_time ** 3)
        
        # 4阶：加入snap项（角加加速度的变化率）
        if max_order >= 4 and recent_history_length >= 10:
            # 估计snap（jerk的变化率）
            roll_snap = torch.zeros(num_envs, device=current_roll.device)
            pitch_snap = torch.zeros(num_envs, device=current_pitch.device)
            
            if recent_history_length >= 12:
                # 使用多个时间点的jerk估计snap
                roll_ang_vel_1 = roll_ang_vel_history[-5:-2]
                roll_ang_vel_2 = roll_ang_vel_history[-8:-5]
                roll_acc_1 = (roll_ang_vel_1[2] - roll_ang_vel_1[0]) / (2 * dt)
                roll_acc_2 = (roll_ang_vel_2[2] - roll_ang_vel_2[0]) / (2 * dt)
                roll_jerk_1 = (roll_acc_1 - roll_acc_2) / (3 * dt)
                
                roll_ang_vel_3 = roll_ang_vel_history[-11:-8]
                roll_acc_3 = (roll_ang_vel_3[2] - roll_ang_vel_3[0]) / (2 * dt)
                roll_jerk_2 = (roll_acc_2 - roll_acc_3) / (3 * dt)
                
                roll_snap = (roll_jerk_1 - roll_jerk_2) / (3 * dt)
                
                # 对pitch做同样的处理
                pitch_ang_vel_1 = pitch_ang_vel_history[-5:-2]
                pitch_ang_vel_2 = pitch_ang_vel_history[-8:-5]
                pitch_acc_1 = (pitch_ang_vel_1[2] - pitch_ang_vel_1[0]) / (2 * dt)
                pitch_acc_2 = (pitch_ang_vel_2[2] - pitch_ang_vel_2[0]) / (2 * dt)
                pitch_jerk_1 = (pitch_acc_1 - pitch_acc_2) / (3 * dt)
                
                pitch_ang_vel_3 = pitch_ang_vel_history[-11:-8]
                pitch_acc_3 = (pitch_ang_vel_3[2] - pitch_ang_vel_3[0]) / (2 * dt)
                pitch_jerk_2 = (pitch_acc_2 - pitch_acc_3) / (3 * dt)
                
                pitch_snap = (pitch_jerk_1 - pitch_jerk_2) / (3 * dt)
            
            roll_taylor = roll_taylor + (1.0 / 24.0) * roll_snap * (prediction_time ** 4)
            pitch_taylor = pitch_taylor + (1.0 / 24.0) * pitch_snap * (prediction_time ** 4)
        
        # 方法3：使用历史姿态数据直接外推（大幅改进：支持傅里叶分析、高阶多项式、样条插值）
        # 如果历史姿态数据可用，使用多种高级拟合方法预测未来姿态
        roll_from_history_fit = None
        pitch_from_history_fit = None
        
        if quat_history is not None and recent_history_length >= 5:
            # 关键改进：所有平台的运动都是完全一致的，只需要计算一次（环境0），然后广播到所有环境
            # 提取更多历史姿态数据（最多使用50个点，大幅增加）
            max_fit_points = min(50, recent_history_length)  # 从30增加到50
            recent_quat = quat_history[-max_fit_points:, 0:1, :]  # 只使用环境0的数据 [num_points, 1, 4]
            history_roll_fit = torch.zeros(recent_quat.shape[0], device=current_roll.device)  # [num_points]
            history_pitch_fit = torch.zeros(recent_quat.shape[0], device=current_pitch.device)  # [num_points]
            
            for i in range(recent_quat.shape[0]):
                roll_i, pitch_i, _ = euler_xyz_from_quat(recent_quat[i])
                history_roll_fit[i] = roll_i[0]  # 只提取环境0
                history_pitch_fit[i] = pitch_i[0]  # 只提取环境0
            
            # 时间点：从0开始，每个点间隔dt
            time_points = torch.arange(recent_quat.shape[0], device=current_roll.device, dtype=torch.float32) * dt
            future_time = time_points[-1] + prediction_time
            
            # 加权：最近的点权重更大
            weights_fit = torch.exp(-0.2 * (recent_quat.shape[0] - 1 - torch.arange(recent_quat.shape[0], device=current_roll.device)))
            weights_fit = weights_fit / weights_fit.sum()
            
            # 只计算一次（环境0），然后广播到所有环境
            y_roll = history_roll_fit  # [num_points]
            y_pitch = history_pitch_fit  # [num_points]
            
            # 方法3.1：傅里叶级数拟合（适合周期性运动）- 大幅改进
            # y(t) = a₀ + Σ(aₙ*cos(nωt) + bₙ*sin(nωt))
            roll_fourier = None
            pitch_fourier = None
            roll_fourier_error = float('inf')
            pitch_fourier_error = float('inf')
            
            if recent_quat.shape[0] >= 10:  # 至少需要10个点进行傅里叶拟合
                try:
                    # ========== 关键改进1：使用FFT更准确地估计基础频率 ==========
                    # 对历史数据进行FFT，找到主要频率成分
                    # 注意：使用全局的torch模块，避免作用域问题
                    
                    # 使用去趋势后的数据进行FFT（去除线性趋势，突出周期性）
                    # 对roll和pitch分别处理
                    y_roll_detrend = y_roll - torch.linspace(y_roll[0], y_roll[-1], len(y_roll), device=y_roll.device)
                    y_pitch_detrend = y_pitch - torch.linspace(y_pitch[0], y_pitch[-1], len(y_pitch), device=y_pitch.device)
                    
                    # FFT（使用全局torch模块，确保正确访问）
                    # 兼容不同PyTorch版本
                    try:
                        # 尝试使用torch.fft（PyTorch 1.8+）
                        fft_roll = torch.fft.rfft(y_roll_detrend)
                        fft_pitch = torch.fft.rfft(y_pitch_detrend)
                        freqs = torch.fft.rfftfreq(len(y_roll_detrend), dt)
                    except AttributeError:
                        # 回退：使用自相关方法估计频率（不依赖torch.fft）
                        # 使用自相关找到主要周期
                        def autocorr(x, max_lag=None):
                            """计算自相关函数"""
                            if max_lag is None:
                                max_lag = len(x) // 2
                            x_centered = x - x.mean()
                            autocorr_vals = []
                            for lag in range(1, min(max_lag, len(x) // 2)):
                                corr = (x_centered[:-lag] * x_centered[lag:]).mean()
                                autocorr_vals.append(corr)
                            return torch.tensor(autocorr_vals, device=x.device) if autocorr_vals else torch.tensor([], device=x.device)
                        
                        # 使用自相关找到主要周期
                        autocorr_roll = autocorr(y_roll_detrend)
                        autocorr_pitch = autocorr(y_pitch_detrend)
                        
                        if len(autocorr_roll) > 0:
                            # 找到第一个峰值（主要周期）
                            peaks_roll = []
                            for i in range(1, len(autocorr_roll) - 1):
                                if autocorr_roll[i] > autocorr_roll[i-1] and autocorr_roll[i] > autocorr_roll[i+1]:
                                    peaks_roll.append(i)
                            
                            if peaks_roll:
                                period_roll = peaks_roll[0] * dt
                                omega_base_roll = 2 * torch.pi / max(period_roll, dt * 2)
                            else:
                                T_est = time_points[-1] - time_points[0]
                                omega_base_roll = 2 * torch.pi / max(T_est, dt * 2)
                        else:
                            T_est = time_points[-1] - time_points[0]
                            omega_base_roll = 2 * torch.pi / max(T_est, dt * 2)
                        
                        if len(autocorr_pitch) > 0:
                            peaks_pitch = []
                            for i in range(1, len(autocorr_pitch) - 1):
                                if autocorr_pitch[i] > autocorr_pitch[i-1] and autocorr_pitch[i] > autocorr_pitch[i+1]:
                                    peaks_pitch.append(i)
                            
                            if peaks_pitch:
                                period_pitch = peaks_pitch[0] * dt
                                omega_base_pitch = 2 * torch.pi / max(period_pitch, dt * 2)
                            else:
                                T_est = time_points[-1] - time_points[0]
                                omega_base_pitch = 2 * torch.pi / max(T_est, dt * 2)
                        else:
                            T_est = time_points[-1] - time_points[0]
                            omega_base_pitch = 2 * torch.pi / max(T_est, dt * 2)
                        
                        # 设置默认值，跳过FFT部分
                        top_indices_roll = torch.tensor([], dtype=torch.long, device=current_roll.device)
                        top_indices_pitch = torch.tensor([], dtype=torch.long, device=current_pitch.device)
                    
                    # 如果FFT成功，找到主要频率
                    if 'freqs' in locals() and len(freqs) > 0:
                        # 找到主要频率（功率最大的频率，排除DC分量）
                        power_roll = torch.abs(fft_roll[1:]) ** 2  # 排除DC分量
                        power_pitch = torch.abs(fft_pitch[1:]) ** 2
                        
                        # 找到前3个主要频率
                        top_k = min(3, len(power_roll))
                        if top_k > 0:
                            _, top_indices_roll = torch.topk(power_roll, top_k)
                            _, top_indices_pitch = torch.topk(power_pitch, top_k)
                        else:
                            top_indices_roll = torch.tensor([], dtype=torch.long, device=current_roll.device)
                            top_indices_pitch = torch.tensor([], dtype=torch.long, device=current_pitch.device)
                        
                        # 使用主要频率的平均值作为基础频率（更准确）
                        if len(top_indices_roll) > 0:
                            main_freq_roll = freqs[top_indices_roll[0] + 1].item()  # +1因为排除了DC分量
                            omega_base_roll = 2 * torch.pi * main_freq_roll
                        else:
                            # 回退：使用总时间跨度估计
                            T_est = time_points[-1] - time_points[0]
                            omega_base_roll = 2 * torch.pi / max(T_est, dt * 2)
                        
                        if len(top_indices_pitch) > 0:
                            main_freq_pitch = freqs[top_indices_pitch[0] + 1].item()
                            omega_base_pitch = 2 * torch.pi * main_freq_pitch
                        else:
                            T_est = time_points[-1] - time_points[0]
                            omega_base_pitch = 2 * torch.pi / max(T_est, dt * 2)
                    
                    # ========== 关键改进2：使用多频率成分（不仅限于谐波） ==========
                    # 使用前3个主要频率，每个频率使用1-2个谐波
                    # 使用roll和pitch的主要频率的并集，构建统一的基函数
                    # 确保所有变量都被正确初始化
                    if 'top_indices_roll' not in locals():
                        top_indices_roll = torch.tensor([], dtype=torch.long, device=current_roll.device)
                    if 'top_indices_pitch' not in locals():
                        top_indices_pitch = torch.tensor([], dtype=torch.long, device=current_pitch.device)
                    if 'omega_base_roll' not in locals():
                        T_est = time_points[-1] - time_points[0]
                        omega_base_roll = 2 * torch.pi / max(T_est, dt * 2)
                    if 'omega_base_pitch' not in locals():
                        T_est = time_points[-1] - time_points[0]
                        omega_base_pitch = 2 * torch.pi / max(T_est, dt * 2)
                    
                    # 确保top_indices_roll和top_indices_pitch是tensor
                    if not isinstance(top_indices_roll, torch.Tensor):
                        top_indices_roll = torch.tensor([], dtype=torch.long, device=current_roll.device)
                    if not isinstance(top_indices_pitch, torch.Tensor):
                        top_indices_pitch = torch.tensor([], dtype=torch.long, device=current_pitch.device)
                    
                    if len(top_indices_roll) > 0 or len(top_indices_pitch) > 0:
                        all_top_indices = torch.cat([top_indices_roll, top_indices_pitch]) if len(top_indices_roll) > 0 and len(top_indices_pitch) > 0 else (top_indices_roll if len(top_indices_roll) > 0 else top_indices_pitch)
                        unique_indices = torch.unique(all_top_indices)
                        num_main_freqs = min(3, len(unique_indices))
                    else:
                        # 如果没有找到主要频率，使用基础频率
                        unique_indices = torch.tensor([], dtype=torch.long, device=current_roll.device)
                        num_main_freqs = 1  # 使用单一基础频率
                    harmonics_per_freq = 2  # 每个主要频率使用2个谐波
                    
                    # 构建多频率傅里叶基函数矩阵（使用统一的频率集合）
                    num_terms = 1 + num_main_freqs * harmonics_per_freq * 2  # 1个DC + 每个频率2个谐波（sin+cos）
                    X_fourier = torch.ones(recent_quat.shape[0], num_terms, device=current_roll.device)
                    
                    col_idx = 1
                    for freq_idx in range(num_main_freqs):
                        if freq_idx < len(unique_indices) and 'freqs' in locals() and len(freqs) > unique_indices[freq_idx] + 1:
                            freq_idx_actual = unique_indices[freq_idx].item()
                            freq = freqs[freq_idx_actual + 1].item()  # +1因为排除了DC分量
                            omega = 2 * torch.pi * freq
                        else:
                            # 回退：使用基础频率的倍数
                            omega = omega_base_roll * (freq_idx + 1)
                        
                        # 对每个主要频率使用多个谐波
                        for h in range(1, harmonics_per_freq + 1):
                            omega_n = h * omega
                            X_fourier[:, col_idx] = torch.cos(omega_n * time_points)
                            col_idx += 1
                            X_fourier[:, col_idx] = torch.sin(omega_n * time_points)
                            col_idx += 1
                    
                    # 确保列数正确
                    if col_idx < num_terms:
                        X_fourier = X_fourier[:, :col_idx]
                        num_terms = col_idx
                    
                    # 加权最小二乘法求解傅里叶系数
                    W = torch.diag(weights_fit)
                    XTWX_fourier = X_fourier.t() @ W @ X_fourier
                    XTWy_roll_fourier = X_fourier.t() @ W @ y_roll
                    XTWy_pitch_fourier = X_fourier.t() @ W @ y_pitch
                    
                    # 添加正则化项（防止过拟合）
                    reg_lambda = 1e-6
                    XTWX_fourier_reg = XTWX_fourier + reg_lambda * torch.eye(num_terms, device=current_roll.device)
                    
                    coeffs_roll_fourier = torch.linalg.solve(XTWX_fourier_reg, XTWy_roll_fourier)
                    coeffs_pitch_fourier = torch.linalg.solve(XTWX_fourier_reg, XTWy_pitch_fourier)
                    
                    # ========== 关键改进3：计算拟合误差，用于后续权重调整 ==========
                    # 计算历史拟合误差
                    y_roll_fitted = X_fourier @ coeffs_roll_fourier
                    y_pitch_fitted = X_fourier @ coeffs_pitch_fourier
                    roll_fourier_error = torch.sqrt(torch.mean((y_roll - y_roll_fitted) ** 2)).item()
                    pitch_fourier_error = torch.sqrt(torch.mean((y_pitch - y_pitch_fitted) ** 2)).item()
                    
                    # 预测未来：使用傅里叶级数（使用相同的频率集合）
                    X_future_fourier = torch.ones(1, num_terms, device=current_roll.device)
                    col_idx = 1
                    for freq_idx in range(num_main_freqs):
                        if freq_idx < len(unique_indices) and 'freqs' in locals() and len(freqs) > unique_indices[freq_idx] + 1:
                            freq_idx_actual = unique_indices[freq_idx].item()
                            freq = freqs[freq_idx_actual + 1].item()
                            omega = 2 * torch.pi * freq
                        else:
                            omega = omega_base_roll * (freq_idx + 1)
                        
                        for h in range(1, harmonics_per_freq + 1):
                            omega_n = h * omega
                            X_future_fourier[0, col_idx] = torch.cos(omega_n * future_time)
                            col_idx += 1
                            X_future_fourier[0, col_idx] = torch.sin(omega_n * future_time)
                            col_idx += 1
                    
                    if col_idx < num_terms:
                        X_future_fourier = X_future_fourier[:, :col_idx]
                    
                    roll_fourier = (coeffs_roll_fourier * X_future_fourier[0]).sum()
                    pitch_fourier = (coeffs_pitch_fourier * X_future_fourier[0]).sum()
                except Exception as e:
                    roll_fourier = None
                    pitch_fourier = None
                    roll_fourier_error = float('inf')
                    pitch_fourier_error = float('inf')
            
            # 方法3.2：高阶多项式拟合（支持1-5阶）
            roll_poly = None
            pitch_poly = None
            
            # 根据历史数据长度自适应选择多项式阶数
            if recent_quat.shape[0] >= 25:
                poly_order = 5  # 历史数据非常充足，使用5阶
            elif recent_quat.shape[0] >= 18:
                poly_order = 4  # 历史数据充足，使用4阶
            elif recent_quat.shape[0] >= 12:
                poly_order = 3  # 历史数据中等，使用3阶
            elif recent_quat.shape[0] >= 8:
                poly_order = 2  # 历史数据较少，使用2阶
            else:
                poly_order = 1  # 历史数据很少，使用1阶（线性）
            
            try:
                # 构建范德蒙德矩阵
                X_poly = torch.ones(recent_quat.shape[0], poly_order + 1, device=current_roll.device)
                for order in range(1, poly_order + 1):
                    X_poly[:, order] = time_points ** order
                
                # 加权最小二乘法
                W = torch.diag(weights_fit)
                XTWX_poly = X_poly.t() @ W @ X_poly
                XTWy_roll_poly = X_poly.t() @ W @ y_roll
                XTWy_pitch_poly = X_poly.t() @ W @ y_pitch
                
                coeffs_roll_poly = torch.linalg.solve(XTWX_poly, XTWy_roll_poly)
                coeffs_pitch_poly = torch.linalg.solve(XTWX_poly, XTWy_pitch_poly)
                
                # 预测未来
                future_time_powers = torch.tensor([future_time ** i for i in range(poly_order + 1)], device=current_roll.device)
                roll_poly = (coeffs_roll_poly * future_time_powers).sum()
                pitch_poly = (coeffs_pitch_poly * future_time_powers).sum()
            except:
                roll_poly = None
                pitch_poly = None
            
            # 方法3.3：样条插值（使用最近几个点进行三次样条）
            roll_spline = None
            pitch_spline = None
            
            if recent_quat.shape[0] >= 4:  # 至少需要4个点进行三次样条
                try:
                    # 使用最近8个点进行样条插值
                    n_spline = min(8, recent_quat.shape[0])
                    t_spline = time_points[-n_spline:]
                    y_roll_spline = y_roll[-n_spline:]
                    y_pitch_spline = y_pitch[-n_spline:]
                    
                    # 简化的样条外推：使用最后两个点的线性外推（更稳定）
                    # 或者使用最后三个点的二次外推
                    if n_spline >= 3:
                        # 使用最后三个点进行二次外推
                        t_last3 = t_spline[-3:]
                        y_roll_last3 = y_roll_spline[-3:]
                        y_pitch_last3 = y_pitch_spline[-3:]
                        
                        # 二次拟合：y = at² + bt + c
                        A = torch.stack([
                            t_last3**2, t_last3, torch.ones(3, device=current_roll.device)
                        ], dim=1)
                        
                        coeffs_roll_spline = torch.linalg.solve(A, y_roll_last3)
                        coeffs_pitch_spline = torch.linalg.solve(A, y_pitch_last3)
                        
                        roll_spline = coeffs_roll_spline[0] * future_time**2 + coeffs_roll_spline[1] * future_time + coeffs_roll_spline[2]
                        pitch_spline = coeffs_pitch_spline[0] * future_time**2 + coeffs_pitch_spline[1] * future_time + coeffs_pitch_spline[2]
                    else:
                        # 使用最后两个点进行线性外推
                        roll_spline = y_roll_spline[-1] + (y_roll_spline[-1] - y_roll_spline[-2]) / (t_spline[-1] - t_spline[-2]) * prediction_time
                        pitch_spline = y_pitch_spline[-1] + (y_pitch_spline[-1] - y_pitch_spline[-2]) / (t_spline[-1] - t_spline[-2]) * prediction_time
                except:
                    roll_spline = None
                    pitch_spline = None
                
            # ========== 关键改进4：根据拟合误差动态调整权重 ==========
            # 融合多种方法：根据可用性、可靠性和拟合误差加权
            # 优先级：傅里叶（如果可用且误差小）> 高阶多项式 > 样条 > 线性
            predictions_roll = []
            predictions_pitch = []
            weights_method = []
            
            # 计算多项式拟合误差（如果可用）
            roll_poly_error = float('inf')
            pitch_poly_error = float('inf')
            if roll_poly is not None:
                try:
                    # 使用多项式预测历史点，计算误差
                    future_time_powers = torch.tensor([t ** i for i in range(poly_order + 1) for t in time_points], 
                                                      device=current_roll.device).reshape(len(time_points), poly_order + 1)
                    y_roll_poly_fitted = (coeffs_roll_poly.unsqueeze(0) * future_time_powers).sum(dim=1)
                    y_pitch_poly_fitted = (coeffs_pitch_poly.unsqueeze(0) * future_time_powers).sum(dim=1)
                    roll_poly_error = torch.sqrt(torch.mean((y_roll - y_roll_poly_fitted) ** 2)).item()
                    pitch_poly_error = torch.sqrt(torch.mean((y_pitch - y_pitch_poly_fitted) ** 2)).item()
                except:
                    pass
            
            # 计算样条拟合误差（如果可用）
            roll_spline_error = float('inf')
            pitch_spline_error = float('inf')
            if roll_spline is not None:
                try:
                    # 使用样条预测历史点，计算误差
                    if n_spline >= 3:
                        A = torch.stack([t_spline**2, t_spline, torch.ones(n_spline, device=current_roll.device)], dim=1)
                        y_roll_spline_fitted = (coeffs_roll_spline.unsqueeze(0) * A).sum(dim=1)
                        y_pitch_spline_fitted = (coeffs_pitch_spline.unsqueeze(0) * A).sum(dim=1)
                        roll_spline_error = torch.sqrt(torch.mean((y_roll_spline - y_roll_spline_fitted) ** 2)).item()
                        pitch_spline_error = torch.sqrt(torch.mean((y_pitch_spline - y_pitch_spline_fitted) ** 2)).item()
                except:
                    pass
            
            # 根据拟合误差计算权重（误差越小，权重越大）
            # 使用逆误差作为权重，但限制在合理范围内
            max_error = 1.0  # 最大可接受误差（rad）
            
            if roll_fourier is not None and torch.isfinite(roll_fourier) and abs(roll_fourier.item()) < 10.0:
                predictions_roll.append(roll_fourier)
                predictions_pitch.append(pitch_fourier)
                # 权重 = 1 / (误差 + 小常数)，归一化后使用
                error_roll = min(roll_fourier_error, max_error)
                error_pitch = min(pitch_fourier_error, max_error)
                weight_fourier = 1.0 / (error_roll + error_pitch + 0.01)  # 0.01防止除零
                weights_method.append(weight_fourier)
            
            if roll_poly is not None and torch.isfinite(roll_poly) and abs(roll_poly.item()) < 10.0:
                predictions_roll.append(roll_poly)
                predictions_pitch.append(pitch_poly)
                error_roll = min(roll_poly_error, max_error)
                error_pitch = min(pitch_poly_error, max_error)
                weight_poly = 1.0 / (error_roll + error_pitch + 0.01)
                weights_method.append(weight_poly)
            
            if roll_spline is not None and torch.isfinite(roll_spline) and abs(roll_spline.item()) < 10.0:
                predictions_roll.append(roll_spline)
                predictions_pitch.append(pitch_spline)
                error_roll = min(roll_spline_error, max_error)
                error_pitch = min(pitch_spline_error, max_error)
                weight_spline = 1.0 / (error_roll + error_pitch + 0.01)
                weights_method.append(weight_spline)
            
            # 只计算一次（环境0），然后广播到所有环境
            if len(predictions_roll) > 0:
                # 归一化权重
                weights_method = torch.tensor(weights_method, device=current_roll.device)
                weights_method = weights_method / weights_method.sum()
                
                roll_from_history_fit_single = sum(w * p for w, p in zip(weights_method, predictions_roll))
                pitch_from_history_fit_single = sum(w * p for w, p in zip(weights_method, predictions_pitch))
            else:
                # 如果所有方法都失败，使用简单线性外推
                if recent_quat.shape[0] >= 2:
                    roll_from_history_fit_single = y_roll[-1] + (y_roll[-1] - y_roll[-2]) / dt * prediction_time
                    pitch_from_history_fit_single = y_pitch[-1] + (y_pitch[-1] - y_pitch[-2]) / dt * prediction_time
                else:
                    roll_from_history_fit_single = current_roll[0]  # 只使用环境0
                    pitch_from_history_fit_single = current_pitch[0]  # 只使用环境0
            
            # 广播到所有环境（因为所有平台的运动都是完全一致的）
            roll_from_history_fit = roll_from_history_fit_single.expand(num_envs)
            pitch_from_history_fit = pitch_from_history_fit_single.expand(num_envs)
        
        # 方法4：融合多种外推方法，提高鲁棒性
        # 权重分配：历史拟合（最准确，如果可用）+ Taylor级数（次准确）+ 角速度积分（最稳定）
        if roll_from_history_fit is not None:
            # 如果历史拟合可用，优先使用（50%）+ Taylor级数（30%）+ 角速度积分（20%）
            future_roll = 0.5 * roll_from_history_fit + 0.3 * roll_taylor + 0.2 * roll_from_vel_integration
            future_pitch = 0.5 * pitch_from_history_fit + 0.3 * pitch_taylor + 0.2 * pitch_from_vel_integration
        elif recent_history_length >= 5:
            # 历史数据充足，更信任Taylor级数（60%）+ 角速度积分（40%）
            future_roll = 0.6 * roll_taylor + 0.4 * roll_from_vel_integration
            future_pitch = 0.6 * pitch_taylor + 0.4 * pitch_from_vel_integration
        else:
            # 历史数据较少，更信任角速度积分（更稳定）
            future_roll = 0.4 * roll_taylor + 0.6 * roll_from_vel_integration
            future_pitch = 0.4 * pitch_taylor + 0.6 * pitch_from_vel_integration
        
        # 数值稳定性检查：如果预测值异常大，回退到简单线性外推
        roll_prediction_valid = torch.isfinite(future_roll) & (torch.abs(future_roll) < 10.0)  # 限制在±10弧度内
        pitch_prediction_valid = torch.isfinite(future_pitch) & (torch.abs(future_pitch) < 10.0)
        
        future_roll = torch.where(
            roll_prediction_valid,
            future_roll,
            current_roll + current_roll_ang_vel * prediction_time  # 回退到线性外推
        )
        future_pitch = torch.where(
            pitch_prediction_valid,
            future_pitch,
            current_pitch + current_pitch_ang_vel * prediction_time
        )
        
        return future_roll, future_pitch
    
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
