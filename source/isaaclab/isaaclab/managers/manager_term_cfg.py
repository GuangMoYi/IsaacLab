# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.noise import NoiseCfg

from .scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from .action_manager import ActionTerm
    from .command_manager import CommandTerm
    from .manager_base import ManagerTermBase
    from .recorder_manager import RecorderTerm


@configclass
class ManagerTermBaseCfg:
    """Configuration for a manager term."""

    func: Callable | ManagerTermBase = MISSING
    """The function or class to be called for the term.

    The function must take the environment object as the first argument.
    The remaining arguments are specified in the :attr:`params` attribute.

    It also supports `callable classes`_, i.e. classes that implement the :meth:`__call__`
    method. In this case, the class should inherit from the :class:`ManagerTermBase` class
    and implement the required methods.

    .. _`callable classes`: https://docs.python.org/3/reference/datamodel.html#object.__call__
    """

    params: dict[str, Any | SceneEntityCfg] = dict()
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict.

    .. note::
        If the value is a :class:`SceneEntityCfg` object, the manager will query the scene entity
        from the :class:`InteractiveScene` and process the entity's joints and bodies as specified
        in the :class:`SceneEntityCfg` object.
    """


##
# Recorder manager.
##


@configclass
class RecorderTermCfg:
    """Configuration for an recorder term."""

    class_type: type[RecorderTerm] = MISSING
    """The associated recorder term class.

    The class should inherit from :class:`isaaclab.managers.action_manager.RecorderTerm`.
    """


##
# Action manager.
##


@configclass
class ActionTermCfg:
    """Configuration for an action term."""

    class_type: type[ActionTerm] = MISSING
    """The associated action term class.

    The class should inherit from :class:`isaaclab.managers.action_manager.ActionTerm`.
    """

    asset_name: str = MISSING
    """The name of the scene entity.

    This is the name defined in the scene configuration file. See the :class:`InteractiveSceneCfg`
    class for more details.
    """

    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""

    clip: dict[str, tuple] | None = None
    """Clip range for the action (dict of regex expressions). Defaults to None."""


##
# Command manager.
##


@configclass
class CommandTermCfg:
    """Configuration for a command generator term."""

    class_type: type[CommandTerm] = MISSING
    """The associated command term class to use.

    The class should inherit from :class:`isaaclab.managers.command_manager.CommandTerm`.
    """

    resampling_time_range: tuple[float, float] = MISSING
    """Time before commands are changed [s]."""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""


##
# Curriculum manager.
##


@configclass
class CurriculumTermCfg(ManagerTermBaseCfg):
    """Configuration for a curriculum term."""

    func: Callable[..., float | dict[str, float] | None] = MISSING
    """The name of the function to be called.

    This function should take the environment object, environment indices
    and any other parameters as input and return the curriculum state for
    logging purposes. If the function returns None, the curriculum state
    is not logged.
    """


##
# Observation manager.
##


@configclass
class ObservationTermCfg(ManagerTermBaseCfg):
    """Configuration for an observation term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the observation signal as torch float tensors of
    shape (num_envs, obs_term_dim).
    """

    modifiers: list[ModifierCfg] | None = None
    """The list of data modifiers to apply to the observation in order. Defaults to None,
    in which case no modifications will be applied.

    Modifiers are applied in the order they are specified in the list. They can be stateless
    or stateful, and can be used to apply transformations to the observation data. For example,
    a modifier can be used to normalize the observation data or to apply a rolling average.

    For more information on modifiers, see the :class:`~isaaclab.utils.modifiers.ModifierCfg` class.
    """

    noise: NoiseCfg | None = None
    """The noise to add to the observation. Defaults to None, in which case no noise is added."""

    clip: tuple[float, float] | None = None
    """The clipping range for the observation after adding noise. Defaults to None,
    in which case no clipping is applied."""

    scale: tuple[float, ...] | float | None = None
    """The scale to apply to the observation after clipping. Defaults to None,
    in which case no scaling is applied (same as setting scale to :obj:`1`).

    We leverage PyTorch broadcasting to scale the observation tensor with the provided value. If a tuple is provided,
    please make sure the length of the tuple matches the dimensions of the tensor outputted from the term.
    """

    history_length: int = 0
    """Number of past observations to store in the observation buffers. Defaults to 0, meaning no history.

    Observation history initializes to empty, but is filled with the first append after reset or initialization. Subsequent history
    only adds a single entry to the history buffer. If flatten_history_dim is set to True, the source data of shape
    (N, H, D, ...) where N is the batch dimension and H is the history length will be reshaped to a 2D tensor of shape
    (N, H*D*...). Otherwise, the data will be returned as is.
    """

    flatten_history_dim: bool = True
    """Whether or not the observation manager should flatten history-based observation terms to a 2D (N, D) tensor.
    Defaults to True."""


@configclass
class ObservationGroupCfg:
    """Configuration for an observation group."""

    concatenate_terms: bool = True
    """Whether to concatenate the observation terms in the group. Defaults to True.

    If true, the observation terms in the group are concatenated along the last dimension.
    Otherwise, they are kept separate and returned as a dictionary.

    If the observation group contains terms of different dimensions, it must be set to False.
    """

    enable_corruption: bool = False
    """Whether to enable corruption for the observation group. Defaults to False.

    If true, the observation terms in the group are corrupted by adding noise (if specified).
    Otherwise, no corruption is applied.
    """

    history_length: int | None = None
    """Number of past observation to store in the observation buffers for all observation terms in group.

    This parameter will override :attr:`ObservationTermCfg.history_length` if set. Defaults to None. If None, each
    terms history will be controlled on a per term basis. See :class:`ObservationTermCfg` for details on history_length
    implementation.
    """

    flatten_history_dim: bool = True
    """Flag to flatten history-based observation terms to a 2D (num_env, D) tensor for all observation terms in group.
    Defaults to True.

    This parameter will override all :attr:`ObservationTermCfg.flatten_history_dim` in the group if
    ObservationGroupCfg.history_length is set.
    """


##
# Event manager
##


@configclass
class EventTermCfg(ManagerTermBaseCfg):
    """Configuration for a event term."""

    func: Callable[..., None] = MISSING
    """  指定要调用的函数
    这个函数接收环境对象、环境索引和其他参数作为输入

    当事件触发时，这个函数会被执行

    必须指定一个有效的函数，否则会报错(MISSING表示必须提供)
    """

    mode: str = MISSING
    """  定义事件触发的模式
    必须指定一个有效的模式字符串

    不同的模式决定了事件如何被触发和执行

    保留模式"interval"有特殊处理方式
    """

    interval_range_s: tuple[float, float] | None = None
    """  定义事件触发的时间间隔范围(秒)

    仅在模式为"interval"时使用

    指定一个时间范围(如(5.0, 10.0))，系统会在这个范围内均匀采样确定触发间隔

    如果为None，则不会基于时间间隔触发
    """

    is_global_time: bool = False
    """  控制时间随机化是否在所有环境实例间共享

    仅在模式为"interval"时使用

    True: 所有环境实例使用相同的时间间隔

    False(默认): 每个环境实例独立采样时间间隔
    """

    min_step_count_between_reset: int = 0
    """  定义在"reset"模式下，两次事件触发间的最小步数

    仅在模式为"reset"时使用

    设置为0表示每次调用manager时都会触发事件

    大于0的值可以防止事件触发过于频繁，提高性能
    """


##
# Reward manager.
##


@configclass
class RewardTermCfg(ManagerTermBaseCfg):
    """Configuration for a reward term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the reward signals as torch float tensors of
    shape (num_envs,).
    """

    weight: float = MISSING
    """The weight of the reward term.

    This is multiplied with the reward term's value to compute the final
    reward.

    Note:
        If the weight is zero, the reward term is ignored.
    """


##
# Termination manager.
##


@configclass
class TerminationTermCfg(ManagerTermBaseCfg):
    """Configuration for a termination term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the termination signals as torch boolean tensors of
    shape (num_envs,).
    """

    time_out: bool = False
    """Whether the termination term contributes towards episodic timeouts. Defaults to False.

    Note:
        These usually correspond to tasks that have a fixed time limit.
    """
