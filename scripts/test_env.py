import os
import pathlib
import random
from isaacgym import gymapi

gym = gymapi.acquire_gym()

compute_device_id = 0
graphics_device_id = 0

sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# use gpu pipeline
# sim_params.use_gpu_pipeline = True

sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!

# create the ground plane
gym.add_ground(sim, plane_params)

## Loading Assets
base_path = pathlib.Path(__file__).parent.parent.resolve()
asset_root =  os.path.join(base_path, "assets")
asset_file = "open_manipulator/open_manipulator_robot.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())


spacing = 0.5  
lower = gymapi.Vec3(-spacing, -spacing, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

num_env_per_row = 32
num_envs = num_env_per_row*num_env_per_row

envs = []
actor_handles = []

for i in range(num_envs):
    env = gym.create_env(sim, lower, upper, num_env_per_row)
    envs.append(env)
    actor_handle = gym.create_actor(env, asset, pose, "OpenManipulatorRobot", 0, 1)
    actor_handles.append(actor_handle)

cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 1, 0))

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
