# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import pathlib
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch, unscale, quat_apply, tensor_clamp, torch_rand_float
from glob import glob
from manipulation_gym.utils.misc import tprint
from .base.vec_task import VecTask
from termcolor import cprint


class OpenManipulatorMove(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):

        cprint('\n\nopen_manipulator_move.py: INIT START ...\n',
               'green', attrs=['bold'])

        self.config = config
        # before calling init in VecTask, need to do
        # 1. setup randomization
        self._setup_domain_rand_config(config['env']['randomization'])
        # 2. setup privileged information
        self._setup_priv_option_config(config['env']['privInfo'])
        # 4. setup reward
        self._setup_reward_config(config['env']['reward'])

        self.aggregate_mode = self.config['env']['aggregateMode']
        self.up_axis = 'z'
        self.evaluate = self.config['on_evaluation']
        self.priv_info_dict = {
            'obj_position': (0, 3),
            'obj_scale': (3, 4),
            'obj_mass': (4, 5),
            'obj_friction': (5, 6),
            'obj_com': (6, 9),
        }

        super().__init__(config, sim_device, graphics_device_id, headless)

        self.debug_viz = self.config['env']['enableDebugVis']
        self.max_episode_length = self.config['env']['episodeLength']
        self.dt = self.sim_params.dt

        if self.viewer:
            cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(
            self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)

        # create some wrapper tensors for different slices
        self.omx_default_dof_pos = torch.zeros(
            self.num_omx_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(
            net_contact_forces).view(self.num_envs, -1, 3)
        self.omx_dof_state = self.dof_state.view(
            self.num_envs, -1, 2)[:, :self.num_omx_dofs]
        self.omx_dof_pos = self.omx_dof_state[..., 0]
        self.omx_dof_vel = self.omx_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(
            rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(
            actor_root_state_tensor).view(-1, 13)

        # [num_envs, 13]                    -> position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        print(f"root_state_tensor: {self.root_state_tensor.shape}")
        # [num_envs*num_dofs_per_env, 2]    -> 2 are for position and velocity there are 16 envs and 6 dof_state per env (4 joints, 2 for gripper)
        print(f"dof_state: {self.dof_state.shape}")
        # [num_envs, num_rigid_bodies, 13]  -> 16 env, 7 rigid bodies, 13 for state
        print(f"rigid_body_states: {self.rigid_body_states.shape}")
        # [num_envs, num_rigid_bodies, 3]   -> 6 env, 7 rigid bodies, 13 for state
        print(f"contact_forces: {self.contact_forces.shape}")
        # [num_envs, num_dofs_per_env, 2]   -> 16, 6, 2
        print(f"omx_dof_state: {self.omx_dof_state.shape}")
        # [num_envs, num_dofs_per_env]      -> 16, 6
        print(f"omx_dof_pos: {self.omx_dof_pos.shape}")
        # [num_envs, num_dofs_per_env]      -> 16, 6
        print(f"omx_dof_vel: {self.omx_dof_vel.shape}")
        # since there are no other actors present in the env therfore, omx variables are same as total variables, but this is general method

        self._refresh_gym()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_joint_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_joint_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # apply random forces parameters (We will apply random forces to links, mainly to end-effector)
        self.force_scale = self.config['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.config['env'].get(
            'randomForceProbScalar', 0.0)
        self.force_decay = self.config['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.config['env'].get(
            'forceDecayInterval', 0.08)
        self.force_decay = to_torch(
            self.force_decay, dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        # useful buffers
        self.init_pose_buf = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        # num_actions defined in config
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.torques = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [
            int, float], 'assume p_gain and d_gain are only scalars'
        self.p_gain = torch.ones((self.num_envs, self.num_dofs),
                                 device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_dofs),
                                 device=self.device, dtype=torch.float) * self.d_gain

        # debug and understanding statistics
        self.env_timeout_counter = to_torch(np.zeros(len(self.envs))).long().to(
            self.device)  # max 10 (10000 envs)
        self.stat_sum_rewards = 0
        self.stat_sum_episode_length = 0
        self.stat_sum_torques = 0
        self.env_evaluated = 0
        self.max_evaluate_envs = 500000

        # [num_envs]
        print(
            f"env_timeout_counter: {self.env_timeout_counter, self.env_timeout_counter.shape}")

        cprint('\nopen_manipulator_move.py: INIT COMPLETE\n\n',
               'green', attrs=['bold'])

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._create_assets()

        # set omx dof properties
        self.num_omx_dofs = self.gym.get_asset_dof_count(self.omx_asset)
        omx_dof_props = self.gym.get_asset_dof_properties(self.omx_asset)

        self.omx_dof_lower_limits = []
        self.omx_dof_upper_limits = []

        for i in range(self.num_omx_dofs):
            self.omx_dof_lower_limits.append(omx_dof_props['lower'][i])
            self.omx_dof_upper_limits.append(omx_dof_props['upper'][i])
            omx_dof_props['effort'][i] = 0.5
            if self.torque_control:
                omx_dof_props['stiffness'][i] = 0.
                omx_dof_props['damping'][i] = 0.
                omx_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            else:
                omx_dof_props['stiffness'][i] = self.config['env']['controller']['pgain']
                omx_dof_props['damping'][i] = self.config['env']['controller']['dgain']
            omx_dof_props['friction'][i] = 0.01
            omx_dof_props['armature'][i] = 0.001

        self.omx_dof_lower_limits = to_torch(
            self.omx_dof_lower_limits, device=self.device)
        self.omx_dof_upper_limits = to_torch(
            self.omx_dof_upper_limits, device=self.device)

        omx_pose = self._init_actor_pose()

        # compute aggregate size
        self.num_omx_bodies = self.gym.get_asset_rigid_body_count(
            self.omx_asset)
        self.num_omx_shapes = self.gym.get_asset_rigid_shape_count(
            self.omx_asset)
        max_agg_bodies = self.num_omx_bodies + 2
        max_agg_shapes = self.num_omx_shapes + 2

        self.envs = []

        self.omx_indices = []

        print(f"num_omx_dofs: {self.num_omx_dofs}")
        print(f"num_omx_bodies: {self.num_omx_bodies}")
        print(f"num_omx_shapes: {self.num_omx_shapes}")

        print(f"Setting up {num_envs} envs")

        for i in range(num_envs):

            # create env instance
            env_handle = self.gym.create_env(
                self.sim, lower, upper, num_per_row)
            if self.aggregate_mode:
                self.gym.begin_aggregate(
                    env_handle, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            omx_actor = self.gym.create_actor(
                env_handle, self.omx_asset, omx_pose, 'omx', i, -1, 0)
            self.gym.set_actor_dof_properties(
                env_handle, omx_actor, omx_dof_props)
            omx_idx = self.gym.get_actor_index(
                env_handle, omx_actor, gymapi.DOMAIN_SIM)
            self.omx_indices.append(omx_idx)

            if self.aggregate_mode:
                self.gym.end_aggregate(env_handle)

            self.envs.append(env_handle)

        self.omx_indices = to_torch(
            self.omx_indices, dtype=torch.long, device=self.device)

        print(f"omx_indices: {self.omx_indices}")

    def reset_idx(self, env_ids):
        if self.randomize_mass:
            lower, upper = self.randomize_mass_lower, self.randomize_mass_upper

            # pass
            # Randomise the mass of the links (add noise)

        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(
                    env_ids), self.num_dofs),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(
                    env_ids), self.num_dofs),
                device=self.device).squeeze(1)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        num_scales = len(self.randomize_scale_list)

        omx_indices = self.omx_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(
                self.prev_joint_targets), gymtorch.unwrap_tensor(omx_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(
            self.dof_state), gymtorch.unwrap_tensor(omx_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def compute_observations(self):
        self._refresh_gym()

        # cur_obs_buf = [num_envs, 1, num_actions*2]
        # A = (curr_joint_positions + noise) -> unscale (x, lower, upper) -> represent q (joint positons)
        # B = cur_joint_targets[:num_actions] -> represent actions (joint torques)
        # cur_obs_buf = [A,B] -> [num_envs, 1, num_actions*2]
        # get observation history [num_envs, 80, num_obs//3] num_obs = num_actions*2*3 (2 for concating actions to joint positions for observation, 3 for time)
        # insert current observation into observation history at last position (delete the first position)
        
        joint_noise_matrix_obs = (torch.rand(self.omx_dof_pos.shape)
                              * 2.0 - 1.0) * self.joint_noise_scale
        cur_obs_buf = unscale(
            joint_noise_matrix_obs.to(
                self.device) + self.omx_dof_pos, self.omx_dof_lower_limits, self.omx_dof_upper_limits
        ).clone()[:,:self.num_actions].unsqueeze(1)
        cur_tar_buf = self.cur_joint_targets[:, None, :self.num_actions]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)

        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # print(cur_obs_buf.shape, prev_obs_buf.shape, self.obs_buf_lag_history.shape)

        # refill the buffers of envs which are reset
        # obs_history[:num_actions] = unscale(joint_pos, lower, upper) 
        # obs_history[num_action:] = joint_pos ***This should actually be a_{t-1} but at t=0 a_{-1} does not exist
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:self.num_actions] = unscale(
            self.omx_dof_pos[at_reset_env_ids], self.omx_dof_lower_limits,
            self.omx_dof_upper_limits
        ).clone()[:,:self.num_actions].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, self.num_actions:] = self.omx_dof_pos[at_reset_env_ids][:,:self.num_actions].unsqueeze(1)

        # select last 3 time-steps of obs_history (3, num_actions*2) and then flatten them to (num_actions*2*3)=num_obs 
        # put this into self.obs_buf (it has shape [num_envs, num_obs])
        t_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone() 
        self.obs_buf[:, :t_buf.shape[1]] = t_buf
        self.at_reset_buf[at_reset_env_ids] = 0

        # put last 30 observations into self.proprio_hist_buf
        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()

        # self._update_priv_buf(env_id=range(self.num_envs),name='obj_position', value=self.object_pos.clone())

    def compute_reward(self, actions):
        # pose diff penalty
        pose_diff_penalty = (
            (self.omx_dof_pos - self.init_pose_buf) ** 2).sum(-1)
        # work and torque penalty
        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = ((self.torques * self.dof_vel_finite_diff).sum(-1)) ** 2
        pose_diff_pscale = self.pose_diff_penalty_scale
        torque_pscale = self.torque_penalty_scale
        work_pscale = self.work_penalty_scale

        self.rew_buf[:] = compute_omx_reward(
            pose_diff_penalty, pose_diff_pscale,
            torque_penalty, torque_pscale,
            work_penalty, work_pscale,
        )
        self.reset_buf[:] = self.check_termination()
        self.extras['pose_diff_penalty'] = pose_diff_penalty.mean()
        self.extras['work_done'] = work_penalty.mean()
        self.extras['torques'] = torque_penalty.mean()

        if self.evaluate:
            finished_episode_mask = self.reset_buf == 1
            self.stat_sum_rewards += self.rew_buf.sum()
            self.stat_sum_torques += self.torques.abs().sum()
            self.stat_sum_episode_length += (self.reset_buf == 0).sum()
            self.env_evaluated += (self.reset_buf == 1).sum()
            self.env_timeout_counter[finished_episode_mask] += 1
            info = f'progress {self.env_evaluated} / {self.max_evaluate_envs} | ' \
                   f'reward: {self.stat_sum_rewards / self.env_evaluated:.2f} | ' \
                   f'eps length: {self.stat_sum_episode_length / self.env_evaluated:.2f} | ' \
                   f'command torque: {self.stat_sum_torques / self.stat_sum_episode_length:.2f}'
            tprint(info)
            if self.env_evaluated >= self.max_evaluate_envs:
                exit()

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        self.compute_reward(self.actions)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()

        # self.debug_viz is false

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i],
                           to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i],
                           to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i],
                           to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [
                                   p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):

        self.actions = actions.clone().to(self.device)

        # print(self.prev_targets.shape, self.actions.shape)

        targets = self.prev_joint_targets.clone().to(self.device)
        targets[:, :self.num_actions] += 1 / 24 * self.actions
        # targets = torch.cat((self.prev_targets[:, :self.num_actions] + 1 / 24 * self.actions, self.prev_targets[:, self.num_actions:]), dim=1)

        self.cur_joint_targets[:] = tensor_clamp(
            targets, self.omx_dof_lower_limits, self.omx_dof_upper_limits)
        self.prev_joint_targets[:] = self.cur_joint_targets.clone()

        # if self.force_scale > 0.0:
        #     self.rb_forces *= torch.pow(self.force_decay,
        #                                 self.dt / self.force_decay_interval)
        #     # apply new forces
        #     obj_mass = to_torch(
        #         [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for
        #          env in self.envs], device=self.device)
        #     prob = self.random_force_prob_scalar
        #     force_indices = (torch.less(torch.rand(
        #         self.num_envs, device=self.device), prob)).nonzero()
        #     self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
        #         self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * obj_mass[force_indices, None] * self.force_scale
        #     self.gym.apply_rigid_body_force_tensors(
        #         self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

    def reset(self):
        super().reset()
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(
            self.rl_device)
        return self.obs_dict

    def step(self, actions):

        print(f"omm.py step -> super()")
        super().step(actions)


        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(
            self.rl_device)
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def update_low_level_control(self):
        previous_dof_pos = self.omx_dof_pos.clone()
        self._refresh_gym()
        if self.torque_control:
            dof_pos = self.omx_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff = dof_vel.clone()
            torques = self.p_gain * (self.cur_joint_targets - dof_pos) - self.d_gain * dof_vel
            self.torques = torch.clip(torques, -0.5, 0.5).clone()
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques))
        else:
            print("position control")
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.cur_joint_targets))

    def check_termination(self):
        # resets = torch.logical_or(
        #     torch.less(object_pos[:, -1], self.reset_z_threshold),
        #     torch.greater_equal(self.progress_buf, self.max_episode_length),
        # )

        resets = torch.greater_equal(self.progress_buf, self.max_episode_length)
        return resets

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config['randomizeMass']
        self.randomize_mass_lower = rand_config['randomizeMassLower']
        self.randomize_mass_upper = rand_config['randomizeMassUpper']
        self.randomize_com = rand_config['randomizeCOM']
        self.randomize_com_lower = rand_config['randomizeCOMLower']
        self.randomize_com_upper = rand_config['randomizeCOMUpper']
        self.randomize_friction = rand_config['randomizeFriction']
        self.randomize_friction_lower = rand_config['randomizeFrictionLower']
        self.randomize_friction_upper = rand_config['randomizeFrictionUpper']
        self.randomize_scale = rand_config['randomizeScale']
        self.scale_list_init = rand_config['scaleListInit']
        self.randomize_scale_list = rand_config['randomizeScaleList']
        self.randomize_scale_lower = rand_config['randomizeScaleLower']
        self.randomize_scale_upper = rand_config['randomizeScaleUpper']
        self.randomize_pd_gains = rand_config['randomizePDGains']
        self.randomize_p_gain_lower = rand_config['randomizePGainLower']
        self.randomize_p_gain_upper = rand_config['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_config['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_config['randomizeDGainUpper']
        self.joint_noise_scale = rand_config['jointNoiseScale']

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config['enableObjPos']
        self.enable_priv_obj_mass = p_config['enableObjMass']
        self.enable_priv_obj_scale = p_config['enableObjScale']
        self.enable_priv_obj_com = p_config['enableObjCOM']
        self.enable_priv_obj_friction = p_config['enableObjFriction']

    def _update_priv_buf(self, env_id, name, value, lower=None, upper=None):
        # normalize to -1, 1
        s, e = self.priv_info_dict[name]
        if eval(f'self.enable_priv_{name}'):
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            if type(lower) is list or upper is list:
                lower = to_torch(lower, dtype=torch.float, device=self.device)
                upper = to_torch(upper, dtype=torch.float, device=self.device)
            if lower is not None and upper is not None:
                value = (2.0 * value - upper - lower) / (upper - lower)
            self.priv_info_buf[env_id, s:e] = value
        else:
            self.priv_info_buf[env_id, s:e] = 0

    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        self.prop_hist_len = self.config['env']['propHistoryLen']
        self.num_env_factors = self.config['env']['privInfoDim']
        self.priv_info_buf = torch.zeros(
            (num_envs, self.num_env_factors), device=self.device, dtype=torch.float)
        self.proprio_hist_buf = torch.zeros(
            (num_envs, self.prop_hist_len, self.num_actions*2), device=self.device, dtype=torch.float)

    def _setup_reward_config(self, r_config):
        self.angvel_clip_min = r_config['angvelClipMin']
        self.angvel_clip_max = r_config['angvelClipMax']
        self.rotate_reward_scale = r_config['rotateRewardScale']
        self.object_linvel_penalty_scale = r_config['objLinvelPenaltyScale']
        self.pose_diff_penalty_scale = r_config['poseDiffPenaltyScale']
        self.torque_penalty_scale = r_config['torquePenaltyScale']
        self.work_penalty_scale = r_config['workPenaltyScale']

    def _create_assets(self):

        print("Creating assets")

        # object file to asset
        asset_root = os.path.join(pathlib.Path(
            __file__).parent.parent.parent.resolve(), "assets")

        omx_asset_file = self.config['env']['asset']['open_manipulator_asset']
        # load omx asset
        omx_asset_options = gymapi.AssetOptions()
        omx_asset_options.flip_visual_attachments = False
        omx_asset_options.fix_base_link = True
        omx_asset_options.collapse_fixed_joints = True
        omx_asset_options.disable_gravity = True
        omx_asset_options.thickness = 0.001
        omx_asset_options.angular_damping = 0.01

        if self.torque_control:
            omx_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        else:
            omx_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.omx_asset = self.gym.load_asset(
            self.sim, asset_root, omx_asset_file, omx_asset_options)

    def _init_actor_pose(self):
        omx_start_pose = gymapi.Transform()
        omx_start_pose.p = gymapi.Vec3(0, 0, 0.5)
        omx_start_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 1, 0), -np.pi / 2) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi / 2)
        return omx_start_pose


def compute_omx_reward(
    pose_diff_penalty, pose_diff_penalty_scale: float,
    torque_penalty, torque_pscale: float,
    work_penalty, work_pscale: float,
):
    reward = 0
    # Distance from the hand to the object
    reward = reward + pose_diff_penalty * pose_diff_penalty_scale
    reward = reward + torque_penalty * torque_pscale
    reward = reward + work_penalty * work_pscale
    return reward
