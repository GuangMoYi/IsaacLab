# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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
    For example, if the asset at prim path expression "/World/envs/env_.*/Object" has a child
    with the path "/World/envs/env_.*/Object/mesh", then the relative child path should be "mesh" or
    "/mesh".

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
    stage = omni.usd.get_context().get_stage()
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


def randomize_rigid_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    recompute_inertia: bool = True,
):
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
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

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

    # get the current masses of the bodies (num_assets, num_bodies)
    masses = asset.root_physx_view.get_masses()

    # apply randomization on default values
    # this is to make sure when calling the function multiple times, the randomization is applied on the
    # default values and not the previously randomized values
    masses[env_ids[:, None], body_ids] = asset.data.default_mass[env_ids[:, None], body_ids].clone()

    # sample from the given range
    # note: we modify the masses in-place for all environments
    #   however, the setter takes care that only the masses of the specified environments are modified
    masses = _randomize_prop_by_op(
        masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )

    # set the mass into the physics simulation
    asset.root_physx_view.set_masses(masses, env_ids)

    # recompute inertia tensors if needed
    if recompute_inertia:
        # compute the ratios of the new masses to the initial masses
        ratios = masses[env_ids[:, None], body_ids] / asset.data.default_mass[env_ids[:, None], body_ids]
        # scale the inertia tensors by the the ratios
        # since mass randomization is done on default values, we can use the default inertia tensors
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            # inertia has shape: (num_envs, num_bodies, 9) for articulation
            inertias[env_ids[:, None], body_ids] = (
                asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            )
        else:
            # inertia has shape: (num_envs, 9) for rigid object
            inertias[env_ids] = asset.data.default_inertia[env_ids] * ratios
        # set the inertia tensors into the physics simulation
        asset.root_physx_view.set_inertias(inertias, env_ids)


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
    coms[:, body_ids, :3] += rand_samples

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


def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the actuator gains in an articulation by adding, scaling, or setting random values.

    This function allows randomizing the actuator stiffness and damping gains.

    The function samples random values from the given distribution parameters and applies the operation to the joint properties.
    It then sets the values into the actuator models. If the distribution parameters are not provided for a particular property,
    the function does not modify the property.

    .. tip::
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        In such cases, it is recommended to use this function only during the initialization of the environment.
    """
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    def randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
        return _randomize_prop_by_op(
            data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
        )

    # Loop through actuators and randomize gains
    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            # we take all the joints of the actuator
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            else:
                global_indices = torch.tensor(actuator.joint_indices, device=asset.device)
        elif isinstance(actuator.joint_indices, slice):
            # we take the joints defined in the asset config
            global_indices = actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            # we take the intersection of the actuator joints and the asset config joints
            actuator_joint_indices = torch.tensor(actuator.joint_indices, device=asset.device)
            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            # the indices of the joints in the actuator that have to be randomized
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue
            # maps actuator indices that have to be randomized to global joint indices
            global_indices = actuator_joint_indices[actuator_indices]
        # Randomize stiffness
        if stiffness_distribution_params is not None:
            stiffness = actuator.stiffness[env_ids].clone()
            stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()
            randomize(stiffness, stiffness_distribution_params)
            actuator.stiffness[env_ids] = stiffness
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
        # Randomize damping
        if damping_distribution_params is not None:
            damping = actuator.damping[env_ids].clone()
            damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
            randomize(damping, damping_distribution_params)
            actuator.damping[env_ids] = damping
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


def randomize_joint_parameters(
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
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # sample joint properties from the given ranges and set into the physics simulation
    # joint friction coefficient
    if friction_distribution_params is not None:
        friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_friction_coeff.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_friction_coefficient_to_sim(
            friction_coeff[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
        )

    # joint armature
    if armature_distribution_params is not None:
        armature = _randomize_prop_by_op(
            asset.data.default_joint_armature.clone(),
            armature_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_armature_to_sim(armature[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids)

    # joint position limits
    if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
        joint_pos_limits = asset.data.default_joint_pos_limits.clone()
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
                "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater than"
                " upper joint limits. Please check the distribution parameters for the joint position limits."
            )
        # set the position limits into the physics simulation
        asset.write_joint_position_limit_to_sim(
            joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
        )


def randomize_fixed_tendon_parameters(
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
    """Randomize the simulated fixed tendon parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the fixed tendon parameters of the asset.
    These correspond to the physics engine tendon properties that affect the joint behavior.

    The function samples random values from the given distribution parameters and applies the operation to the tendon properties.
    It then sets the values into the physics simulation. If the distribution parameters are not provided for a
    particular property, the function does not modify the property.

    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.fixed_tendon_ids == slice(None):
        tendon_ids = slice(None)  # for optimization purposes
    else:
        tendon_ids = torch.tensor(asset_cfg.fixed_tendon_ids, dtype=torch.int, device=asset.device)

    # sample tendon properties from the given ranges and set into the physics simulation
    # stiffness
    if stiffness_distribution_params is not None:
        stiffness = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_stiffness.clone(),
            stiffness_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_stiffness(stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # damping
    if damping_distribution_params is not None:
        damping = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_damping.clone(),
            damping_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_damping(damping[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # limit stiffness
    if limit_stiffness_distribution_params is not None:
        limit_stiffness = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_limit_stiffness.clone(),
            limit_stiffness_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_limit_stiffness(limit_stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # position limits
    if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
        limit = asset.data.default_fixed_tendon_pos_limits.clone()
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
                "Randomization term 'randomize_fixed_tendon_parameters' is setting lower tendon limits that are greater"
                " than upper tendon limits."
            )
        asset.set_fixed_tendon_position_limit(tendon_limits, tendon_ids, env_ids)

    # rest length
    if rest_length_distribution_params is not None:
        rest_length = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_rest_length.clone(),
            rest_length_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_rest_length(rest_length[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # offset
    if offset_distribution_params is not None:
        offset = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_offset.clone(),
            offset_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_offset(offset[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # write the fixed tendon properties into the simulation
    asset.write_fixed_tendon_properties_to_sim(tendon_ids, env_ids)


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

    # 读取当前速度
    vel_w = asset.data.root_vel_w[env_ids]
    # 采样随机速度
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # 应用速度到物理仿真
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

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

    
# GMY
def move_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],  # 必填参数
    asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
):
    """
    函数作用：在设置的位置和角度范围内，周期性地正弦扰动刚体的位置和姿态
    输入参数：
        env: ManagerBasedEnv, 环境实例
        env_ids: torch.Tensor, 环境ID
        position_range: dict[str, tuple[float, float]]  设置位置/角度变化范围（振幅） {"x":(), "y":(), "z":(), "roll":(), "pitch":(), "yaw":()}
        asset_cfg: SceneEntityCfg = SceneEntityCfg("platform")  作用对象
    """
    range_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

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
    # rot_angles = (rot_angles + math.pi) % (2 * math.pi) - math.pi           # 归一化到 [-pi, pi]： 这是因为有些角度2*pi没法处理
    
    current_pos = asset.data.root_pos_w[env_ids]                            # 读取当前位置
    relative_pos = current_pos - initial_pos

    # 拼接位置和角度
    pose = torch.cat([relative_pos, rot_angles], dim=1)

    # 当前时间
    time = 0.25 * env._sim_step_counter * env.physics_dt
    t = torch.tensor(time, device=asset.device)

    # 读取正弦扰动范围
    range_list = [position_range.get(key, (0.0, 0.0)) for key in range_keys]
    ranges = torch.tensor(range_list, device=asset.device)  # [6, 2]，分别是 (振幅, 相位)

    A = ranges[:, 0]  # 振幅
    phi = ranges[:, 1]  # 相位
    T_tmie = 0.2        # 周期 (s)
    omega = 2 * math.pi / T_tmie  # 角频率

    delta = A * torch.sin(omega * t + phi)  
    
    # 平移扰动
    delta_pos = delta[:3].unsqueeze(0)  # [N, 3]
    new_pos = pose[:,:3] + delta_pos

    # 欧拉角扰动
    delta_euler = delta[3:].unsqueeze(0) # [N, 3]
    roll = delta_euler[:, 0]
    pitch = delta_euler[:, 1]
    yaw = delta_euler[:, 2]
    delta_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw) # [4, N]
    delta_quat = math_utils.quat_mul(delta_quat, initial_quat)

    # 新姿态 = 初始姿态 * delta_quat
    new_quat = math_utils.quat_mul(delta_quat, q_rel)

    # 拼接 root_pose: [x, y, z, qw, qx, qy, qz]
    root_pose = torch.cat([delta_pos, delta_quat], dim=1)  # [N, 7]
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)

from isaaclab.envs.mdp.vessels import frigate, semisub, supply
def move_acceleration(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    if not hasattr(env, '_acc_tensor'):
        import pandas as pd
        import os

        file_name = "data_now"
        experiment_index = 1
        file_path = os.path.expanduser(f'~/IsaacLab/save_data/{file_name}.xlsx')
        df = pd.read_excel(file_path, skiprows=2)

        cols_per_experiment = 6
        start_col = (experiment_index - 1) * cols_per_experiment
        end_col = experiment_index * cols_per_experiment

        exp_df = df.iloc[:, start_col:end_col]
        exp_df.columns = ['X_acc', 'Y_acc', 'Z_acc', 'X_ang_acc', 'Y_ang_acc', 'Z_ang_acc']
        # 将角加速度从度转换为弧度 (最后三列是角加速度)
        exp_df = exp_df.copy()
        exp_df[['X_ang_acc', 'Y_ang_acc', 'Z_ang_acc']] = exp_df[['X_ang_acc', 'Y_ang_acc', 'Z_ang_acc']] * (math.pi / 180)

        acc_tensor = torch.tensor(exp_df.values, dtype=torch.float32)
        env._acc_tensor = acc_tensor.to(env.device)  # 保存到环境对象中




    # 运行的不错的： frigate、supply、ROVzefakkel、remus100（有三个角度，但是有个角度要把船翻了）、shipClarke83、tanker
    # frigate('headingAutopilot', 10.0, 30.0)
    # semisub('DPcontrol', 10.0, 10.0, 40.0, 0.5, 190.0)
    # supply('DPcontrol', 5.0, 5.0, 28.64, 0.5, 20.0)
    # DSRV('depthAutopilot', 60.0)
    # otter('headingAutopilot', 100.0, 0.3, -30.0, 200.0)
    # ROVzefakkel('headingAutopilot', 3.0, 100.0)
    # remus100('depthHeadingAutopilot', 30, 50, 1525, 0.5, 170)
    # remus100('depthHeadingAutopilot', 0.5, 30, 300, 0.5, 170)
    # shipClarke83('headingAutopilot', -20.0, 70, 8, 6, 0.7, 0.5, 10.0, 1e5)
    # tanker('headingAutopilot', -20, 0.5, 150, 20, 80)
    # torpedo('depthHeadingAutopilot', 30, 50, 1525, 0.5, 170)
    if not hasattr(env, '_vehicle_dict'):
        env._vehicle_dict = {}
    for i in env_ids.tolist():
        if i not in env._vehicle_dict:
            env._vehicle_dict[i] = semisub('DPcontrol', 0.0, 0.0, 0, 10, 0.1)

    if not hasattr(env, '_vehicle_dict1'):
        env._vehicle_dict1 = {}
    for i in env_ids.tolist():
        if i not in env._vehicle_dict1:
            env._vehicle_dict1[i] = semisub('DPcontrol', 0.0, 0.0, 0, 10, 0.1)

    # 初始化位姿参考
    if not hasattr(env, '_initial_platform_rot'):
        setattr(env, '_initial_platform_rot', asset.data.root_quat_w.clone())
    initial_quat = getattr(env, '_initial_platform_rot')[env_ids]
    if not hasattr(env, '_initial_platform_pos'):
        setattr(env, '_initial_platform_pos', asset.data.root_pos_w.clone())
    initial_pos = getattr(env, '_initial_platform_pos')[env_ids]

    # 初始化时间记录
    # if not hasattr(env, '_last_acceleration_time'):
    #     setattr(env, '_last_acceleration_time', time.time())

    # # 检查时间间隔
    # current_time = time.time()
    # time_since_last = current_time - getattr(env, '_last_acceleration_time')
    # if time_since_last < 0.02:
    #     return  # 如果时间间隔小于 0.02 秒，直接返回，不执行后续代码

    

    dt = 1 * env.physics_dt
    time_me = (0.25 * env._sim_step_counter)
    print("[INFO] 循环函数次数:", time_me)

    current_quat = asset.data.root_quat_w[env_ids]
    q_rel = math_utils.quat_mul(current_quat, math_utils.quat_conjugate(initial_quat.clone().detach()))
    rot_angles = torch.stack(math_utils.euler_zyx_from_quat(current_quat), dim=1)
    # rot_angles = (rot_angles + math.pi) % (2 * math.pi) - math.pi

    current_pos = asset.data.root_pos_w[env_ids]
    relative_pos = current_pos - initial_pos
    pose = torch.cat([relative_pos, rot_angles], dim=1)

    current_pose = torch.cat([current_pos, rot_angles], dim=1)
            
    # nu_lin = asset.data.root_lin_vel_b[env_ids]
    # nu_ang = asset.data.root_ang_vel_b[env_ids] 
    # nu = torch.cat([nu_lin, nu_ang], dim=-1)
    # nu_dot = batch_get_acceleration(pose, nu, env._vehicle_dict, 0.02, env._vehicle_dict1, time_me)
    # # print("[INFO] 加速度:", nu_dot)
    # lin_acc = nu_dot[:, :3]
    # ang_acc = nu_dot[:, 3:]
    # lin_acc = math_utils.quat_rotate(current_quat, lin_acc)  # 线加速度
    # ang_acc = math_utils.quat_rotate(current_quat, ang_acc)  # 角加速度



    acc = get_acceleration_row(env._acc_tensor, i=int(time_me - 1), N=env_ids.shape[0]).to(asset.device)
    lin_acc = 0 * acc[:, :3]
    ang_acc = 0 * acc[:, 3:]

    if time_me >= 100:
        # # GMY
        # # # # 初始化标志位
        if hasattr(env, '_acceleration_applied') and env._acceleration_applied < 0.005 and env._acceleration_applied !=0:
            setattr(env, '_acceleration_applied', env._acceleration_applied + 0.001)
        elif hasattr(env, '_acceleration_applied') and env._acceleration_applied >= 0.005:
            setattr(env, '_acceleration_applied', 0)

        if not hasattr(env, '_acceleration_applied'):
            setattr(env, '_acceleration_applied', 1)

        acc_get = getattr(env, '_acceleration_applied')
        
        ang_acc[:, 0] = acc_get  # 设置x方向加速度为1
        ang_acc[:, 1] = acc_get
        ang_acc[:, 2] = acc_get

        lin_acc[:, 0] = acc_get  # 设置x方向加速度为1
        lin_acc[:, 1] = acc_get
        lin_acc[:, 2] = acc_get
    if time_me >=100:
        print("!!!!!!!!!!")

    
    # # print("[INFO] 位置:", pose)
    print("[INFO] 角速度:", asset.data.root_ang_vel_w[0,:])
    print("[INFO] 线速度:", asset.data.root_lin_vel_w[0,:])
    print("[INFO] 线加速度:", asset.data.body_lin_acc_w[0, :])
    print("[INFO] 角加速度:", asset.data.body_ang_acc_w[0, :])

    ang_acc_vec = ang_acc.unsqueeze(-1)   # 从 shape [N, 3] 变成 [N, 3, 1]
    # 获取刚体质量和惯性（注意此处假设只有一个body）
    mass = asset.data.default_mass.to(asset.device)[env_ids].unsqueeze(-1)
    inertia = asset.data.default_inertia.to(asset.device)[env_ids]  # [N, 9]
    inertia_mat = inertia.view(-1, 3, 3)                            # 从 shape [N, 9] 变成 [N, 3, 3]

    # print("[INFO] 质量:", mass)
    print("[INFO] inertia:", inertia)
    # 使用全部元素
    force = mass * lin_acc.unsqueeze(1) # +  mass * gravity                   # [N, 1, 3]
    torque = torch.matmul(inertia_mat, ang_acc_vec).squeeze(-1)  # [N, 3]

    # # 只使用对角线元素
    # inertia_diag = torch.diag_embed(inertia[:, [0,4,8]])
    # torque = torch.matmul(inertia_diag, ang_acc_vec).squeeze(-1) 
    print("[INFO] 力矩:", torque)
   
    # GMY
    # print("[INFO] 速度:", torch.cat([asset.data.root_lin_vel_b, asset.data.root_ang_vel_b], dim=-1))
    
    # 更新最后一次计算时间
    # setattr(env, '_last_acceleration_time', current_time)

    asset.set_external_force_and_torque(
        forces=force,
        torques=torque.unsqueeze(1),
        body_ids=[0],             # 默认主刚体索引为0
        env_ids=env_ids,
    )

    # 应用加速度后设置标志位为True
    # setattr(env, '_acceleration_applied', True)

# import pandas as pd
# import os
def get_acceleration_row(acc_tensor: torch.Tensor, i: int, N: int) -> torch.Tensor:
    """
    从张量 acc_tensor 中周期性地提取第 i 行的加速度数据，生成形状为 [N, 6] 的张量。
    
    读取规则：
    - 第一轮：正向读取 acc_tensor[i]
    - 第二轮：反向读取，并取负值 acc_tensor[rev_i] * -1
    - 如此反复，确保物理意义上的加速度变化是有规律的
    - 例如： +3 +2 -5 -1 第二次读取就应该是 +1 +5 -2 -3 以此循环
    
    参数：
    - acc_tensor: 输入张量，形状为 [T, 6]，T 是总的加速度数据数
    - i: 全局索引，递增
    - N: 扩展为 N 行

    返回：
    - acc: [N, 6] 张量，是 acc_tensor 中某一行数据（或其相反数）重复 N 次
    """
    T = acc_tensor.shape[0]  # 数据长度
    cycle_length = 2 * T     # 一个完整的正向+反向周期
    idx_in_cycle = i % cycle_length

    if idx_in_cycle < T:
        # 正向读取
        idx = idx_in_cycle
        acc = acc_tensor[idx]
    else:
        # 反向读取并取负
        idx = cycle_length - 1 - idx_in_cycle
        acc = -acc_tensor[idx]
    # 扩展为 [N, 6]
    return acc.unsqueeze(0).expand(N, -1)



def batch_get_acceleration(pose_batch: torch.Tensor, vel_batch: torch.Tensor, vehicle, sampleTime: float, vehicle1, time_me: float):
    acc_list = []
    for i in range(pose_batch.shape[0]):
        # 如果不同env_id使用不同的vehicle，则使用vehicle[i]
        acc = get_platform_acceleration_from_model(pose_batch[i], vel_batch[i], vehicle[i], sampleTime,  vehicle1[i], time_me)
        acc_list.append(acc)
    return torch.stack(acc_list, dim=0)

def get_platform_acceleration_from_model(eta: torch.Tensor, nu: torch.Tensor, vehicle, sampleTime: float, vehicle1, time_me: float) -> torch.Tensor:
    """
    接收 tensor 输入，内部转 numpy 运算，再返回 tensor 输出。
    """
    # 转为 numpy（确保 detach 和在 CPU 上）
    eta_np = eta.detach().cpu().numpy()
    nu_np = nu.detach().cpu().numpy()

    # Vehicle specific control systems
    if (vehicle.controlMode == 'depthAutopilot'):
        u_control = vehicle.depthAutopilot(eta_np,nu_np,sampleTime)
    elif (vehicle.controlMode == 'headingAutopilot'):
        u_control = vehicle.headingAutopilot(eta_np,nu_np,sampleTime)   
    elif (vehicle.controlMode == 'depthHeadingAutopilot'):
        u_control = vehicle.depthHeadingAutopilot(eta_np,nu_np,sampleTime)             
    elif (vehicle.controlMode == 'DPcontrol'):
        u_control = vehicle.DPcontrol(eta_np,nu_np,sampleTime)
        u_control1 = vehicle1.DPcontrol(vehicle1.eta,vehicle1.nu,sampleTime)   
    elif (vehicle.controlMode == 'stepInput'):
        u_control = vehicle.stepInput(time_me)    
        u_control1 = vehicle1.stepInput(time_me) 

    # 调用 numpy 风格的模型
    nu_dot_np, vehicle.u_actual = vehicle.dynamics(eta_np, nu_np, vehicle.u_actual, u_control, sampleTime, time_me)
    vehicle1_nu_dot, vehicle1.u_actual = vehicle1.dynamics(vehicle1.eta, vehicle1.nu, vehicle1.u_actual, u_control1, sampleTime, time_me)
    vehicle1.nu = vehicle1.nu + sampleTime * vehicle1_nu_dot
    vehicle1.eta = attitudeEuler(vehicle1.eta,vehicle1.nu,sampleTime)

    # print("！！！位置eta！！！", vehicle1.eta)
    # print("！！！速度nu！！！", vehicle1.nu)

    # print("!!!x_d,y_d,z_d!!!   [vehicle1]", vehicle1.x_d, vehicle1.y_d, vehicle1.psi_d)
    # print("!!!x_d,y_d,z_d!!!   [vehicle]", vehicle.x_d, vehicle.y_d, vehicle.psi_d)
    
    # acc = torch.tensor(vehicle.nu, dtype=eta.dtype, device=eta.device)
    # 返回 tensor，保持原始设备和 dtype
    acc = torch.tensor(nu_dot_np, dtype=eta.dtype, device=eta.device)
    return acc

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

    # positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    
    # # platform 
    platform = env.scene["platform"]
    # # positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3] 
    
    # # 固定在平台上方某高度，比如 2 * platform_z + 0.5
    platform_pos = platform.data.root_com_pos_w[env_ids]
    height_offset = 0.5 * platform.cfg.spawn.size[2] + 0.5  # 米

    positions = platform_pos.clone()
    positions[:, 2] += height_offset
    positions[:, 0:2] += rand_samples[:, 0:2] + 10


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
    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


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

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


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


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset the scene to the default state specified in the scene configuration."""
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

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        texture_paths = cfg.params.get("texture_paths")
        event_name = cfg.params.get("event_name")
        texture_rotation = cfg.params.get("texture_rotation", (0.0, 0.0))

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual texture material with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # convert from radians to degrees
        texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

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
        # TODO: Remove the hard-coded "/visuals" part.
        prim_path = f"{asset.cfg.prim_path}/{body_names_regex}/visuals"

        # Create the omni-graph node for the randomization term
        def rep_texture_randomization():
            prims_group = rep.get.prims(path_pattern=prim_path)

            with prims_group:
                rep.randomizer.texture(
                    textures=texture_paths, project_uvw=True, texture_rotate=rep.distribution.uniform(*texture_rotation)
                )

            return prims_group.node

        # Register the event to the replicator
        with rep.trigger.on_custom_event(event_name=event_name):
            rep_texture_randomization()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
    ):
        # import replicator
        import omni.replicator.core as rep

        # only send the event to the replicator
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.
        rep.utils.send_og_event(event_name)


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
        """Initialize the randomization term."""
        super().__init__(cfg, env)

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        colors = cfg.params.get("colors")
        event_name = cfg.params.get("event_name")
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

        # parse the colors into replicator format
        if isinstance(colors, dict):
            # (r, g, b) - low, high --> (low_r, low_g, low_b) and (high_r, high_g, high_b)
            color_low = [colors[key][0] for key in ["r", "g", "b"]]
            color_high = [colors[key][1] for key in ["r", "g", "b"]]
            colors = rep.distribution.uniform(color_low, color_high)
        else:
            colors = list(colors)

        # Create the omni-graph node for the randomization term
        def rep_texture_randomization():
            prims_group = rep.get.prims(path_pattern=mesh_prim_path)

            with prims_group:
                rep.randomizer.color(colors=colors)

            return prims_group.node

        # Register the event to the replicator
        with rep.trigger.on_custom_event(event_name=event_name):
            rep_texture_randomization()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
        mesh_name: str = "",
    ):
        # import replicator
        import omni.replicator.core as rep

        # only send the event to the replicator
        rep.utils.send_og_event(event_name)


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
