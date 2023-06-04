from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import to_torch, quat_rotate, quat_conjugate, quat_mul

import math
import os
import pathlib
import random
import numpy as np

import torch



def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 4)
    return u


def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, ee_vel
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * ee_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Open Manipulator X robot arm environment",
    custom_parameters=[
        {"name": "--num_env", "type": int, "default": 4, "help": "Number of environments"},
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"},
        {"name": "--control_mode", "type": str, "default": "position", "help": "Control mode to use. Options are {position, velocity, effort}"},])

print('\nargs:',args,'\n')

# ----CREATE SIMULATOR----
# get default simulation parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

# use gpu pipeline
sim_params.use_gpu_pipeline = args.use_gpu_pipeline

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# print(sim_params)

# create simulator
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
# sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()
    
# Create viewer
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

if viewer is None:
    print("*** Failed to create viewer")
    quit()

# --------------------------



# ------ LOAD ASSETS -------
# Ground Plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
gym.add_ground(sim, plane_params)

# create table asset
table_dims = gymapi.Vec3(0.6, 0.8, 0.2)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create box asset
box_size = 0.03
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

## Load OpenManipulatorX
base_path = pathlib.Path(__file__).parent.parent.resolve()
asset_root =  os.path.join(base_path, "assets")
omx_asset_file = "open_manipulator/open_manipulator_robot.urdf"

omx_asset_options = gymapi.AssetOptions()
omx_asset_options.fix_base_link = True
# asset_options.flip_visual_attachments = True
# asset_options.armature = 0.01
omx_asset = gym.load_asset(sim, asset_root, omx_asset_file, omx_asset_options)

# -----------------------------


# ------ CONFIGURE ASSETS --------
# configure OpenManipulatorX dofs (joints)
# Fields: ['hasLimits', 'lower', 'upper', 'driveMode', 'velocity', 'effort', 'stiffness', 'damping', 'friction', 'armature']
omx_dof_props = gym.get_asset_dof_properties(omx_asset)
omx_joints_lower_limits = omx_dof_props["lower"]
omx_joints_upper_limits = omx_dof_props["upper"]
omx_joint_range = omx_joints_upper_limits - omx_joints_lower_limits
omx_joints_mid = 0.5 * (omx_joints_upper_limits + omx_joints_lower_limits)

# Drive mode (first 4 joints are rotational, last 2 are gripper prismatic)
assert args.control_mode in {"position", "effort"}, f"Invalid control_mode specified -- options are (position, velocity, effort). Got: {args.control_mode}"

# Stiffness and Damping are used by PD controllers in case of Position and Velocity Control Mode (force_applied = posError * stiffness + velError * damping)
if args.control_mode == "position":
    omx_dof_props["driveMode"][:4].fill(gymapi.DOF_MODE_POS)
    omx_dof_props["stiffness"][:4].fill(400.0)
    omx_dof_props["damping"][:4].fill(40.0)
elif args.control_mode == "velocity":
    omx_dof_props["driveMode"][:4].fill(gymapi.DOF_MODE_VEL)
    omx_dof_props["stiffness"][:4].fill(400.0)
    omx_dof_props["damping"][:4].fill(40.0)
else:
    omx_dof_props["driveMode"][:4].fill(gymapi.DOF_MODE_EFFORT)
    omx_dof_props["stiffness"][:4].fill(0.0)
    omx_dof_props["damping"][:4].fill(0.0)

# gripper Drive Mode is Position
omx_dof_props["driveMode"][4:].fill(gymapi.DOF_MODE_POS) 
omx_dof_props["stiffness"][4:].fill(800.0)
omx_dof_props["damping"][4:].fill(40.0)

# default dof states
omx_num_dofs = gym.get_asset_dof_count(omx_asset)
default_dof_pos = np.zeros(omx_num_dofs, dtype=np.float32)
default_dof_pos[:4] = omx_joints_mid[:4]
default_dof_pos[4:] = omx_joints_upper_limits[4:] # grippers open

# DofState [pos, vel]
default_dof_state = np.zeros(omx_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch
# default_dof_pos_tensor = to_torch(default_dof_pos, device=device) # So that it can be used on GPU

# get link index of end_eff
omx_link_dict = gym.get_asset_rigid_body_dict(omx_asset)
omx_ee_index = omx_link_dict["end_effector_link"]

# -------------------------------------------------------------------


# ---------- CREATE AND CONFIGURE ENV -------------------------------

# Define initial poses of the assets to be initialised in the evn
omx_pose = gymapi.Transform()
omx_pose.p = gymapi.Vec3(0.0, 0.0, table_dims.z)
omx_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5*table_dims.x-0.1, 0.0, 0.5 * table_dims.z)

box_pose = gymapi.Transform()

# Grid of envs
spacing = 1  
lower = gymapi.Vec3(-spacing, -spacing, 0.0)
upper = gymapi.Vec3(spacing, spacing, spacing)
num_envs = args.num_env
num_env_per_row = int(math.sqrt(num_envs))

envs = []
box_idxs = []
ee_idxs = []
init_pos_list = []
init_rot_list = []

for i in range(num_envs):
    env = gym.create_env(sim, lower, upper, num_env_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # add box
    box_pose.p.x = np.random.uniform(low=0.1, high=0.3)
    box_pose.p.y = np.random.uniform(low=-0.2, high=0.2)
    box_pose.p.z = table_dims.z + 0.5 * box_size
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # get global index of box in rigid body state tensor
    box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)


    # Add OpenManipulatorX
    omx_handle = gym.create_actor(env, omx_asset, omx_pose, "OpenManipulatorX", i, 2)

    # set dof properties
    gym.set_actor_dof_properties(env, omx_handle, omx_dof_props)
    gym.set_actor_dof_states(env, omx_handle, default_dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, omx_handle, default_dof_pos)

    # get inital end-eff pose
    ee_handle = gym.find_actor_rigid_body_handle(env, omx_handle, "end_effector_link")
    ee_pose = gym.get_rigid_transform(env, ee_handle)
    init_pos_list.append([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z])
    init_rot_list.append([ee_pose.r.x, ee_pose.r.y, ee_pose.r.z, ee_pose.r.w])

    # get global index of end_effector in rigid body state tensor
    ee_idx = gym.find_actor_rigid_body_index(env, omx_handle, "end_effector_link", gymapi.DOMAIN_SIM)
    ee_idxs.append(ee_idx) 

# Set camera viewing direction [position of camera, view target position]
cam_pos = gymapi.Vec3(2, 2, 2)
cam_target = gymapi.Vec3(0, 0, 0)
middle_env = envs[num_envs // 2 + num_env_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# -----------------------------------------


# ---------- PREPARE TENSORS -----------------
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial end_eff position and orientation tensors (note here we are creating tensor on device (GPU/CPU), thus automatically calculations will be on GPU)
# Main point to clarify is to make calculations on GPU use tensors (with .to(device)) instead of numpy arrays
# Use tensors for float, int etc basically for any variable use tensor
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device) # POS(x,y,z)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device) # QUAT(x,y,z,w)

# end_eff orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# # box corner coords, used to determine grasping yaw
box_half_size = 0.5 * box_size
corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base OpenManipulatorX, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "OpenManipulatorX")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to OpenManipulatorX end_eff
j_eef = jacobian[:, omx_ee_index - 1, :, :4]

# get mass matrix tensor
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "OpenManipulatorX")
mm = gymtorch.wrap_tensor(_massmatrix)
mm = mm[:, :4, :4]          # only need elements corresponding to the OpenManipulatorX arm

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

dof_pos = dof_states[:, 0].view(num_envs, omx_num_dofs, 1)
dof_vel = dof_states[:, 1].view(num_envs, omx_num_dofs, 1)

# Create a tensor noting whether the end_eff should return to the initial position
ee_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)
# ------------------------------------------------------------------------------------

# -------- RUN SIMULATION -----------------
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # # refresh tensors
    # gym.refresh_rigid_body_state_tensor(sim)
    # gym.refresh_dof_state_tensor(sim)
    # gym.refresh_jacobian_tensors(sim)
    # gym.refresh_mass_matrix_tensors(sim)

    # box_pos = rb_states[box_idxs, :3]
    # box_rot = rb_states[box_idxs, 3:7]

    # ee_pos = rb_states[ee_idxs, :3]  #(x,y,z)
    # ee_rot = rb_states[ee_idxs, 3:7] #(x,y,z,w)
    # ee_vel = rb_states[ee_idxs, 7:]  #(x,y,z,x,y,z) linear+angular

    # to_box = box_pos - ee_pos # vector from ee_pos to box position
    # box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    # box_dir = to_box / box_dist
    # box_dot = box_dir @ down_dir.view(3, 1)

    # # how far the end_effector should be from box for grasping
    # grasp_offset = 0.11 if args.control_mode == "position" else 0.10

    # # determine if we're holding the box (grippers are closed and box is near)
    # gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
    # gripped = (gripper_sep < 0.045) & (box_dist < grasp_offset + 0.5 * box_size)

    # yaw_q = cube_grasping_yaw(box_rot, corners)
    # box_yaw_dir = quat_axis(yaw_q, 0)
    # ee_yaw_dir = quat_axis(ee_rot, 0)
    # yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), ee_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

    # # determine if we have reached the initial position; if so allow the end_effector to start moving to the box
    # to_init = init_pos - ee_pos
    # init_dist = torch.norm(to_init, dim=-1)
    # ee_restart = (ee_restart & (init_dist > 0.02)).squeeze(-1)
    # return_to_start = (ee_restart | gripped.squeeze(-1)).unsqueeze(-1)

    # # if end_effector is above box, descend to grasp offset
    # # otherwise, seek a position above the box
    # above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3)).squeeze(-1)
    # grasp_pos = box_pos.clone()
    # grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 2.5)

    # # compute goal position and orientation
    # goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
    # goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))

    # # compute position and orientation error
    # pos_err = goal_pos - ee_pos
    # orn_err = orientation_error(goal_rot, ee_rot)
    # dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

    # # Deploy control based on type
    # if args.control_mode == "position":
    #     pos_action[:, :4] = dof_pos.squeeze(-1)[:, :4] + control_ik(dpose)
    # else:       # osc
    #     effort_action[:, :4] = control_osc(dpose)

    # # gripper actions depend on distance between end_effector and box
    # close_gripper = (box_dist < grasp_offset + 0.02) | gripped
    # # always open the gripper above a certain height, dropping the box and restarting from the beginning
    # ee_restart = ee_restart | (box_pos[:, 2] > 0.6)
    # keep_going = torch.logical_not(ee_restart)
    # close_gripper = close_gripper & keep_going.unsqueeze(-1)
    # grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * num_envs).to(device), torch.Tensor([[0.04, 0.04]] * num_envs).to(device))
    # pos_action[:, 4:6] = grip_acts

    # # Deploy actions
    # gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
# -----------------------------------------

# ------- CLEAN ----------
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
