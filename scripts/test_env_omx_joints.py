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

def deg2rad(deg):
    return (deg*math.pi)/180

def rad2deg(rad):
    return (rad*180)/math.pi

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Open Manipulator X robot arm environment",
    custom_parameters=[
        {"name": "--num_env", "type": int, "default": 4, "help": "Number of environments"},
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"},
        {"name": "--control_mode", "type": str, "default": "position", "help": "Control mode to use. Options are {position, velocity, effort} of joints"},])

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
sim_params.physx.contact_offset = 0.001
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

# ======= Kinematic Structure of OpenManipulator (urdf model) =========

# link_1 ---(joint_1, rev)--> link_2 ---(joint_2, rev)--> link_3 ---(joint_3, rev)--> link_4 ---(joint_4, rev)--> link_5
# link_5 ---(gripper_joint_1, prismatic)--> gipper_link_1
# link_5 ---(gripper_joint_2, prismatic)--> gipper_link_2
# link_5 ---(ee_joint, fixed)--> ee_link (dummy link for directly getting end-eff position)

# Name of Links and Joints and DOFs (movable joints)
# Bodies:
#   0: 'link1'
#   1: 'link2'
#   2: 'link3'
#   3: 'link4'
#   4: 'link5'
#   5: 'end_effector_link'
#   6: 'gripper_link'
#   7: 'gripper_link_sub'
# Joints:
#   0: 'joint1' (Revolute)
#   1: 'joint2' (Revolute)
#   2: 'joint3' (Revolute)
#   3: 'joint4' (Revolute)
#   4: 'end_effector_joint' (Fixed)
#   5: 'gripper' (Prismatic)
#   6: 'gripper_sub' (Prismatic)
# DOFs:
#   0: 'joint1' (Rotation)
#   1: 'joint2' (Rotation)
#   2: 'joint3' (Rotation)
#   3: 'joint4' (Rotation)
#   4: 'gripper' (Translation)
#   5: 'gripper_sub' (Translation)

# DOF Properties ('hasLimits', 'lower', 'upper', 'driveMode', 'velocity', 'effort', 'stiffness', 'damping', 'friction', 'armature')
# [( True, -2.8274333, 2.8274333, 0, 4.8, 1., 0., 0., 0., 0.)
#  ( True, -1.7907078, 1.5707964, 0, 4.8, 1., 0., 0., 0., 0.)
#  ( True, -0.9424778, 1.3823007, 0, 4.8, 1., 0., 0., 0., 0.)
#  ( True, -1.7907078, 2.0420353, 0, 4.8, 1., 0., 0., 0., 0.)
#  ( True, -0.01     , 0.019    , 0, 4.8, 1., 0., 0., 0., 0.)   # gripper (-1cm to 2cm)
#  ( True, -0.01     , 0.019    , 0, 4.8, 1., 0., 0., 0., 0.)]  # gripper (-1cm to 2cm)


# =====================================================================


# ------ CONFIGURE ASSETS --------
# configure OpenManipulatorX dofs (joints)
# Fields: ['hasLimits', 'lower', 'upper', 'driveMode', 'velocity', 'effort', 'stiffness', 'damping', 'friction', 'armature']
omx_dof_props = gym.get_asset_dof_properties(omx_asset)
omx_joints_lower_limits = omx_dof_props["lower"]
omx_joints_upper_limits = omx_dof_props["upper"]
omx_joint_range = omx_joints_upper_limits - omx_joints_lower_limits

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
# default_dof_pos[:4] = deg2rad(np.array([0.0, -45.0, 0.0, 0.0]))
default_dof_pos[:4] = deg2rad(np.array([0.0, 0.0, 0.0, 0.0]))
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

# Grid of envs
spacing = 1  
lower = gymapi.Vec3(-spacing, -spacing, 0.0)
upper = gymapi.Vec3(spacing, spacing, spacing)
num_envs = args.num_env
num_env_per_row = int(math.sqrt(num_envs))

envs = []
ee_idxs = []
init_pos_list = []
init_rot_list = []

for i in range(num_envs):
    env = gym.create_env(sim, lower, upper, num_env_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

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
# init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device) # POS(x,y,z)
# init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device) # QUAT(x,y,z,w)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

dof_pos = dof_states[:, 0].view(num_envs, omx_num_dofs, 1)
# dof_vel = dof_states[:, 1].view(num_envs, omx_num_dofs, 1)

# Create a tensor noting whether the end_eff should return to the initial position
# ee_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
target_joint_pos = torch.zeros_like(dof_pos).squeeze(-1)

# target_joint_pos[0,4] = -0.01
# target_joint_pos[0,5] = -0.01

target_joint_pos[0,4] = 0.019
target_joint_pos[0,5] = 0.019

delta_pos = torch.zeros_like(target_joint_pos)

init_joint_pos = dof_pos.clone().squeeze(-1)


# print(delta_pos, delta_pos.shape) # shape is (num_envs, dofs)

# ------------------------------------------------------------------------------------

current_joint_idx = 4
current_stage = 0
current_joint_dir = [1,-1,1]
np.set_printoptions(precision=2, suppress=True)

# -------- RUN SIMULATION -----------------
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_dof_state_tensor(sim)

    # joint_action = torch.Tensor([0.0, 0.0, 0.0, 0.0]).to(device)
    # gripper_action = torch.Tensor([0.0, 0.0]).to(device)

    # if current_joint_idx<4:

    #     print(f"joint_{current_joint_idx+1}_{current_stage}: {rad2deg(dof_pos[0, :4, 0]).cpu().numpy()}")

    #     joint_action[current_joint_idx] = 10.0*current_joint_dir[current_stage]

    #     if current_stage==0 and (dof_pos[0, current_joint_idx, 0]-init_joint_pos[0,current_joint_idx]) > deg2rad(20):
    #         current_stage = 1
    #     elif current_stage == 1 and (dof_pos[0, current_joint_idx, 0]-init_joint_pos[0,current_joint_idx]) < deg2rad(-20):
    #         current_stage = 2
    #     elif current_stage == 2 and (dof_pos[0, current_joint_idx, 0]-init_joint_pos[0,current_joint_idx]) > deg2rad(0):
    #         current_joint_idx = current_joint_idx + 1
    #         current_stage = 0
    # else:
    #     gripper_gap = 3
    #     print(f"joint_{current_joint_idx+1}_{current_stage}: {dof_pos[0, 4:, 0].cpu().numpy()}")
    #     gripper_action[0] = -0.001
    #     gripper_action[1] = -0.001
        


    

    
    # delta_pos = torch.cat((deg2rad(joint_action),gripper_action),0).repeat(num_envs, 1)
    
    # target_joint_pos = dof_pos.squeeze(-1) + delta_pos

    # target_joint_pos = dof_pos.squeeze(-1)

    # print(delta_pos)
    # print(dof_pos.squeeze(-1))
    # print(target_joint_pos)
    # print("-------------\n")

    # # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(target_joint_pos))

    # gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_pos.squeeze(-1)))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
# -----------------------------------------

# ------- CLEAN ----------
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
