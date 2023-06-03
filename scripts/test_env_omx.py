import os
import pathlib
import random
from isaacgym import gymapi

from isaacgym import gymutil


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Open Manipulator X robot arm environment",
    custom_parameters=[
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])

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

# print(sim_params)

# create simulator
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
# sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# --------------------------


# Create viewer
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(2, 2, 2), gymapi.Vec3(0, 0, 0))

if viewer is None:
    print("*** Failed to create viewer")
    quit()


# Ground Plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
gym.add_ground(sim, plane_params)

## Loading Assets
base_path = pathlib.Path(__file__).parent.parent.resolve()
asset_root =  os.path.join(base_path, "assets")
asset_file = "open_manipulator/open_manipulator_robot.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
# asset_options.armature = 0.01

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


spacing = 0.5  
lower = gymapi.Vec3(-spacing, -spacing, 0.0)
upper = gymapi.Vec3(spacing, spacing, spacing)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

num_env_per_row = 2
num_envs = num_env_per_row*num_env_per_row

envs = []
actor_handles = []

for i in range(num_envs):
    env = gym.create_env(sim, lower, upper, num_env_per_row)
    envs.append(env)
    actor_handle = gym.create_actor(env, asset, pose, "OpenManipulatorRobot", i, 1)
    actor_handles.append(actor_handle)


gym.prepare_sim(sim)    

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)


gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
