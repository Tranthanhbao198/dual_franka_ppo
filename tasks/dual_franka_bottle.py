# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_apply, quat_from_euler_xyz, \
    orientation_error
from isaacgymenvs.tasks.base.vec_task import VecTask


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    # type: (Tensor, float) -> Tensor
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)
    angle = torch.norm(vec, dim=-1, keepdim=True)
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class DualFrankaBottle(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.aggregate_mode = self.cfg["env"].get("aggregateMode", 0)

        # Cấu hình phần thưởng đã được tinh chỉnh
        self.reward_settings = {
            "r_reach_scale": 1.0,
            "r_lift_bottle_scale": 10.0,
            "r_hold_bonus_scale": 0.2,
            "r_lift_cap_scale": 50.0,
            "r_success_bonus": 500.0,
            "p_fall_penalty": -10.0,
            "p_collision_penalty": -2.0,
            "p_time_penalty": -0.01
        }

        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type == "osc", "This task only supports OSC control."

        self.cfg["env"]["numObservations"] = 42
        self.cfg["env"]["numActions"] = 14

        self.states = {}
        self.handles = {}
        self.num_dofs = None
        self.actions = None
        self._init_cubeA_state = None
        self._init_cubeB_state = None
        self._cubeA_state = None
        self._cubeB_state = None
        self._cubeA_id = None
        self._cubeB_id = None
        self.up_axis, self.up_axis_idx = "z", 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.franka_default_dof_pos = to_torch([0.0, -0.3, 0.0, -2.2, 0.0, 2.2, 0.8, 0.04, 0.04], device=self.device)
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], device=self.device)
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)

        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []

        for i in range(self.gym.get_asset_dof_count(franka_asset)):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
            franka_dof_props['damping'][i] = franka_dof_damping[i]
            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)

        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 2.0, 2.0, table_thickness, table_opts)

        bottle_asset_file = "urdf/bottle.urdf"
        cap_asset_file = "urdf/cap.urdf"
        object_opts = gymapi.AssetOptions()
        object_opts.use_mesh_materials = True
        cubeA_asset = self.gym.load_asset(self.sim, asset_root, bottle_asset_file, object_opts)
        cubeB_asset = self.gym.load_asset(self.sim, asset_root, cap_asset_file, object_opts)
        self.cubeA_size = 0.15
        self.cubeB_size = 0.03

        table_z_pos = 0.5
        table_pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, table_z_pos))
        self._table_surface_pos = np.array([0.0, 0.0, table_z_pos + 0.5 * table_thickness])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        robot_dist = 0.6
        franka_left_pose = gymapi.Transform(
            p=gymapi.Vec3(-robot_dist, -robot_dist, self._table_surface_pos[2]),
            r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi / 4)
        )
        franka_right_pose = gymapi.Transform(
            p=gymapi.Vec3(robot_dist, robot_dist, self._table_surface_pos[2]),
            r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -3 * np.pi / 4)
        )

        cubeA_pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, self._table_surface_pos[2] + self.cubeA_size / 2))
        cubeB_pose = gymapi.Transform(
            p=gymapi.Vec3(0.0, 0.0, self._table_surface_pos[2] + self.cubeA_size + self.cubeB_size / 2 + 0.002))

        self.envs = []
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.set_actor_dof_properties(env_ptr, self.gym.create_actor(env_ptr, franka_asset, franka_left_pose,
                                                                             "franka_left", i, 0, 0), franka_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, self.gym.create_actor(env_ptr, franka_asset, franka_right_pose,
                                                                             "franka_right", i, 0, 0), franka_dof_props)
            self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 1, 0)
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_pose, "bottle", i, 2, 0)
            self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_pose, "cap", i, 4, 0)

            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.4, 0.8))
            self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.9, 0.1, 0.1))

            self.envs.append(env_ptr)

        self.init_data()

    def init_data(self):
        env_ptr = self.envs[0]
        self.handles = {
            "grip_left": self.gym.find_actor_rigid_body_handle(env_ptr, 0, "panda_grip_site"),
            "grip_right": self.gym.find_actor_rigid_body_handle(env_ptr, 1, "panda_grip_site"),
        }
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_root_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]

        self._eef_left_state = self._rigid_body_state[:, self.handles["grip_left"], :]
        self._eef_right_state = self._rigid_body_state[:, self.handles["grip_right"], :]

        _jac_l = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "franka_left"))
        _jac_r = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "franka_right"))
        hand_idx_l = self.gym.get_actor_joint_dict(env_ptr, 0)["panda_hand_joint"]
        hand_idx_r = self.gym.get_actor_joint_dict(env_ptr, 1)["panda_hand_joint"]
        self._j_eef_left = _jac_l[:, hand_idx_l, :, :7]
        self._j_eef_right = _jac_r[:, hand_idx_r, :, :7]

        _mm_l = gymtorch.wrap_tensor(self.gym.acquire_mass_matrix_tensor(self.sim, "franka_left"))
        _mm_r = gymtorch.wrap_tensor(self.gym.acquire_mass_matrix_tensor(self.sim, "franka_right"))
        self._mm_left = _mm_l[:, :7, :7]
        self._mm_right = _mm_r[:, :7, :7]

        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        num_actors = self.gym.get_actor_count(env_ptr)
        self._global_indices = torch.arange(self.num_envs * num_actors, device=self.device).view(self.num_envs, -1)

        self.states.update({
            "cubeA_size": torch.full_like(self._eef_left_state[:, 0], self.cubeA_size),
            "cubeB_size": torch.full_like(self._eef_left_state[:, 0], self.cubeB_size),
        })

        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        self._arm_left_idx = slice(0, 7)  # Chỉ 7 khớp tay trái
        self._grip_left_idx = slice(7, 9)  # 2 khớp kẹp trái
        self._arm_right_idx = slice(9, 16)  # Chỉ 7 khớp tay phải
        self._grip_right_idx = slice(16, 18)  # 2 khớp kẹp phải

        self._arm_control_left = self._effort_control[:, self._arm_left_idx]
        self._arm_control_right = self._effort_control[:, self._arm_right_idx]
        self._gripper_control_left = self._pos_control[:, self._grip_left_idx]
        self._gripper_control_right = self._pos_control[:, self._grip_right_idx]

    def _update_states(self):
        self.states.update({
            "q_left": self._q[:, :9],
            "q_right": self._q[:, 9:18],
            "eef_left_pos": self._eef_left_state[:, :3],
            "eef_left_quat": self._eef_left_state[:, 3:7],
            "eef_left_vel": self._eef_left_state[:, 7:],
            "eef_right_pos": self._eef_right_state[:, :3],
            "eef_right_quat": self._eef_right_state[:, 3:7],
            "eef_right_vel": self._eef_right_state[:, 7:],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeB_pos": self._cubeB_state[:, :3],
            "cubeB_quat": self._cubeB_state[:, 3:7],
            "cubeA_to_cubeB_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self._update_states()

    def compute_observations(self):
        self._refresh()
        obs_parts = [
            self.states["cubeA_quat"],
            self.states["cubeA_pos"],
            self.states["cubeA_to_cubeB_pos"],
            self.states["eef_left_pos"],
            self.states["eef_left_quat"],
            self.states["q_left"],
            self.states["eef_right_pos"],
            self.states["eef_right_quat"],
            self.states["q_right"],
        ]
        self.obs_buf = torch.cat(obs_parts, dim=-1)
        return self.obs_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # Tạo trạng thái reset cho vật thể
        if self._init_cubeA_state is None:
            self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
            self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

        self._init_cubeA_state[env_ids, :3] = to_torch([0.0, 0.0, self._table_surface_pos[2] + self.cubeA_size / 2],
                                                       device=self.device)
        self._init_cubeA_state[env_ids, 3:7] = to_torch([0.0, 0.0, 0.0, 1.0], device=self.device)
        self._init_cubeB_state[env_ids, :3] = to_torch(
            [0.0, 0.0, self._table_surface_pos[2] + self.cubeA_size + self.cubeB_size / 2 + 0.002], device=self.device)
        self._init_cubeB_state[env_ids, 3:7] = to_torch([0.0, 0.0, 0.0, 1.0], device=self.device)

        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        cube_indices = torch.unique(torch.cat([
            self._global_indices[env_ids, self._cubeA_id],
            self._global_indices[env_ids, self._cubeB_id]
        ])).to(torch.int32)
        if len(cube_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_state),
                                                         gymtorch.unwrap_tensor(cube_indices), len(cube_indices))

        pos_single = self.franka_default_dof_pos.unsqueeze(0).repeat(len(env_ids), 1)
        pos = torch.cat([pos_single, pos_single.clone()], dim=-1)
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = 0.0
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = 0.0

        franka_indices = self._global_indices[env_ids, :2].flatten().to(torch.int32)
        if len(franka_indices) > 0:
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                  gymtorch.unwrap_tensor(franka_indices), len(franka_indices))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _compute_osc_torques(self, dpose, arm="left"):
        if arm == "left":
            q, qd, mm, j_eef, eef_vel = self._q[:, :7], self._qd[:, :7], self._mm_left, self._j_eef_left, self.states[
                "eef_left_vel"]
        else:
            q, qd, mm, j_eef, eef_vel = self._q[:, 9:16], self._qd[:, 9:16], self._mm_right, self._j_eef_right, \
            self.states["eef_right_vel"]

        mm_inv = torch.inverse(mm)
        m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        u = torch.transpose(j_eef, 1, 2) @ m_eef @ (self.kp * dpose - self.kd * eef_vel).unsqueeze(-1)
        j_eef_inv = m_eef @ j_eef @ mm_inv

        u_null_target = (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi
        u_null = self.kd_null * -qd + self.kp_null * u_null_target
        u_null = mm @ u_null.unsqueeze(-1)

        u += (torch.eye(7, device=self.device) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
        return tensor_clamp(u.squeeze(-1), -self._franka_effort_limits[:7], self._franka_effort_limits[:7])

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        dpose_l = self.actions[:, :6]
        gripper_l_action = self.actions[:, 6]
        dpose_r = self.actions[:, 7:13]
        gripper_r_action = self.actions[:, 13]

        self._arm_control_left[:] = self._compute_osc_torques(dpose=dpose_l * self.cmd_limit, arm="left")
        self._arm_control_right[:] = self._compute_osc_torques(dpose=dpose_r * self.cmd_limit, arm="right")

        open_pos = self.franka_dof_upper_limits[self._grip_left_idx]
        close_pos = self.franka_dof_lower_limits[self._grip_left_idx]
        self._gripper_control_left[:] = torch.where(gripper_l_action.unsqueeze(-1) >= 0, open_pos, close_pos)
        self._gripper_control_right[:] = torch.where(gripper_r_action.unsqueeze(-1) >= 0, open_pos, close_pos)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf,
            self.progress_buf,
            self.states,
            self.reward_settings,
            self.max_episode_length
        )


@torch.jit.script
def compute_franka_reward(reset_buf, progress_buf, states, reward_settings, max_episode_length):
    # type: (Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    eef_l_pos, eef_l_quat = states["eef_left_pos"], states["eef_left_quat"]
    eef_r_pos, eef_r_quat = states["eef_right_pos"], states["eef_right_quat"]
    bottle_pos, bottle_quat = states["cubeA_pos"], states["cubeA_quat"]
    cap_pos = states["cubeB_pos"]

    grasp_thresh = 0.04
    lift_height_goal = 0.05
    separation_goal = 0.03

    # --- Phần thưởng Dẫn đường (Luôn hoạt động) ---
    d_left_to_bottle = torch.norm(eef_l_pos - bottle_pos, dim=-1)
    d_right_to_cap = torch.norm(eef_r_pos - cap_pos, dim=-1)
    reach_reward = torch.exp(-10.0 * d_left_to_bottle) + torch.exp(-10.0 * d_right_to_cap)

    # --- GIAI ĐOẠN 1: Gắp và Nhấc Chai ---
    bottle_height = bottle_pos[:, 2] - reward_settings["table_height"]
    bottle_z_axis_local = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float, device=eef_l_pos.device).repeat(
        eef_l_pos.shape[0], 1)
    bottle_z_axis = quat_apply(bottle_quat, bottle_z_axis_local)

    left_hand_is_grasping = (d_left_to_bottle < grasp_thresh)
    bottle_is_stable = (bottle_z_axis[:, 2] > 0.9)
    bottle_is_lifted_properly = left_hand_is_grasping & bottle_is_stable & (bottle_height > lift_height_goal)

    lift_bottle_reward = torch.where(
        left_hand_is_grasping & bottle_is_stable,
        torch.clamp(bottle_height, min=0.0, max=lift_height_goal) * reward_settings["r_lift_bottle_scale"],
        torch.zeros_like(bottle_height)
    )

    hold_bonus = torch.where(bottle_is_lifted_properly,
                             torch.ones_like(bottle_height) * reward_settings["r_hold_bonus_scale"],
                             torch.zeros_like(bottle_height))

    # --- GIAI ĐOẠN 2: Mở Nắp (Có Điều kiện) ---
    vertical_separation = cap_pos[:, 2] - bottle_pos[:, 2] - (
                0.5 * states["cubeA_size"][0] + 0.5 * states["cubeB_size"][0])
    right_hand_is_ready = (d_right_to_cap < grasp_thresh)
    ready_to_open_cap = bottle_is_lifted_properly & right_hand_is_ready

    lift_cap_reward = torch.where(
        ready_to_open_cap,
        torch.clamp(vertical_separation, min=0.0, max=0.1) * reward_settings["r_lift_cap_scale"],
        torch.zeros_like(vertical_separation)
    )

    world_down_axis = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float, device=eef_l_pos.device).repeat(
        eef_l_pos.shape[0], 1)
    eef_r_z_world = quat_apply(eef_r_quat, bottle_z_axis_local)
    dot_right_down = torch.sum(eef_r_z_world * world_down_axis, dim=-1)
    orient_r_reward = torch.where(ready_to_open_cap, torch.clamp(dot_right_down, min=0),
                                  torch.zeros_like(dot_right_down))

    task_success = ready_to_open_cap & (vertical_separation > separation_goal)
    success_bonus = torch.where(task_success, torch.full_like(d_left_to_bottle, reward_settings["r_success_bonus"]),
                                torch.zeros_like(d_left_to_bottle))

    # --- Hình phạt (Luôn hoạt động) ---
    bottle_is_fallen = (bottle_z_axis[:, 2] < 0.7)
    fall_penalty = torch.where(bottle_is_fallen, torch.full_like(d_left_to_bottle, reward_settings["p_fall_penalty"]),
                               torch.zeros_like(d_left_to_bottle))

    d_hands = torch.norm(eef_l_pos - eef_r_pos, dim=-1)
    collision_penalty = torch.where(d_hands < 0.1, torch.full_like(d_hands, reward_settings["p_collision_penalty"]),
                                    torch.zeros_like(d_hands))

    # --- Tổng hợp Reward ---
    rewards = (reach_reward * reward_settings["r_reach_scale"]) + \
              lift_bottle_reward + \
              hold_bonus + \
              orient_r_reward + \
              lift_cap_reward + \
              success_bonus + \
              fall_penalty + \
              collision_penalty + \
              reward_settings["p_time_penalty"]

    # --- Điều kiện Reset ---
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | task_success | bottle_is_fallen,
                            torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf