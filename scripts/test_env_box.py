from isaacgym import gymapi


headless = False

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
sim_params.use_gpu_pipeline = True

# create simulation 
sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)

# configure and create the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
gym.add_ground(sim, plane_params)

# create asset
asset = gym.create_box(sim, 0.1, 0.1, 0.1)

# Create env
spacing = 0.5  
lower = gymapi.Vec3(-spacing, -spacing, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
env = gym.create_env(sim, lower, upper, 1)

# Add asset to env
actor_handle = gym.create_actor(env, asset, pose, "OpenManipulatorRobot", 0, 1)



# Create and configure viewer and camera
if not headless:
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 1, 0))

gym.prepare_sim(sim)

if headless:
    gym.simulate(sim)
    gym.fetch_results(sim, True)
else:
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

if not headless:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
