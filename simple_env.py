"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
"""

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

import os
import time
import random
import yaml
import torch
import pickle
import numpy as np
import open3d as o3d
import pytorch3d.transforms
from tqdm import tqdm
import cv2
import math
import argparse
from torchvision.transforms.functional import to_pil_image
torch.pi = math.pi


from BallGenerator import BallGenerator
from WeighingDomainInfo import WeighingDomainInfo


def pose7d_to_matrix(pose7d: torch.Tensor):
    matrix = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).repeat(pose7d.shape[0], 1, 1)
    matrix[:, :3, :3] = pytorch3d.transforms.quaternion_to_matrix(pose7d[:, [6, 3, 4, 5]])
    matrix[:, :3, 3] = pose7d[:, :3]

    return matrix

def matrix_to_pose_7d(matrix: torch.Tensor):
    pose_7d = torch.zeros((matrix.shape[0], 7), dtype=torch.float32)
    pose_7d[:, 3:] = pytorch3d.transforms.matrix_to_quaternion(matrix[:, :3, :3])[:, [1, 2, 3, 0]]
    pose_7d[:, :3] = matrix[:, :3, 3]

    return pose_7d

def euler_to_quaternion(roll, pitch, yaw):
    # Abbreviations for the various angular functions
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    q = gymapi.Quat(0,0,0,0)
    q.w = cy * cr * cp + sy * sr * sp
    q.x = cy * sr * cp - sy * cr * sp
    q.y = cy * cr * sp + sy * sr * cp
    q.z = sy * cr * cp - cy * sr * sp

    return q

# Input [x,y,z,w] list with quat
def quaternion_to_euler(q, log=False):
    # roll (x-axis rotation)
    sinr_cosp = +2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = +1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = +2.0 * (q.w * q.y - q.z * q.x)
    if (math.fabs(sinp) >= 1):
        pitch = math.copysign(math.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = +2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)  
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll,pitch,yaw]

class IsaacSim():
    def __init__(self, env_cfg_dict):
        self.env_cfg_dict = env_cfg_dict
        #tool_type : spoon, knife, stir, fork
        self.tool = env_cfg_dict["tool"]
        self.tool_list = np.array(["spoon", "fork", "knife"]) if self.env_cfg_dict["tool"] == "None" else self.env_cfg_dict["tool"]
        # assert self.tool in ["spoon", "knife", "stir", "fork"]
        
        
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # create simulator
        self.env_spacing = 1.5
        self.max_episode_length = 195
        self.asset_root = "urdf"
        self.gravity = -1
        self.create_sim()
        
        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        # keyboard event
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "backward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "turn_right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "turn_left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "turn_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "turn_down")
        
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_1, "move_scoop")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_2, "move_scoop_put")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_3, "move_stir")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_4, "move_fork")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "scoop")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "stir")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "fork")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "cut")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_8, "take_tool")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_7, "put_tool")
        
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_0, "scoop_put")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_9, "change container index")
        
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "gripper_close")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "save")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "to_file")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "quit")

        # Look at the first env
        self.cam_pos = gymapi.Vec3(1., 0, 1.5)
        cam_target = gymapi.Vec3(0, 0, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, cam_target)

        
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        _rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(_rb_state_tensor).view(-1, 13)

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.hand_joint_index, :, :7]
        
        # for trajectory collection
        self.record = []

    def create_sim(self):
        
        # parse arguments
        args = gymutil.parse_arguments(description="Joint control Methods Example", 
                                       custom_parameters=[{'name': '--output', 'type': str, 'default': None}]
                                       )

        args.use_gpu = False
        args.use_gpu_pipeline = False
        self.device = 'cpu'
        #self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_envs = 1
        
        self.output_folder = args.output
        
        if not self.output_folder is None and not os.path.exists(f"./observation/{self.output_folder}"):
            os.mkdir(f"./observation/{self.output_folder}")

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, self.gravity)

        #黏稠度
        sim_params.dt = 1.0/100

        sim_params.substeps = 1

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1

        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.max_depenetration_velocity = 1000

        
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        #self.gym.prepare_sim(self.sim)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))
        
    def add_camera(self, env_ptr):
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1080
        camera_props.height = 720
        camera_props.enable_tensors = True
        cam_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
        self.gym.set_camera_location(cam_handle, env_ptr, gymapi.Vec3(1.5, 0, 1.2), gymapi.Vec3(0, 0, 0))
        self.camera_handles.append(cam_handle)
        
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
    def create_table(self):

        # create table asset
        file_name = 'holder/holder_table.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = True
        asset_options.vhacd_params.resolution = 300000
        self.table_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        # self.table_dims = gymapi.Vec3(0.8, 1, self.default_height)
        # asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        # self.table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, asset_options)
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.5, 0, 0)

    # create bowls & plates
    def create_container(self):
        from matplotlib.colors import to_rgba
        def calculate_dist(pose1, pose2):
            return np.sqrt((pose1.p.x - pose2.p.x) ** 2 + (pose1.p.y - pose2.p.y) ** 2)
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 500000
        if self.env_cfg_dict["containers"] is not None:
            self.container_num = len(self.env_cfg_dict["containers"]) if self.env_cfg_dict["containers"] is not None else 0
            file_name_list = [f'container/{x["type"]}.urdf' for x in self.env_cfg_dict["containers"]] if self.env_cfg_dict["containers"] is not None else None
            self.container_asset = [self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options) for file_name in file_name_list]
        else:
            self.container_num = 0
            file_name_list = []
            self.container_asset = []
        self.containers_pose = []
        self.containers_color = []
        self.min_container_dist = 0.3
        for i in range(self.container_num):
            random_sample = False
            containers_pose = gymapi.Transform()
            containers_pose.r = gymapi.Quat(1, 0, 0, 1) if 'plate' not in self.env_cfg_dict["containers"][i]["type"] else gymapi.Quat(0, 0, 0, 1)
            if self.env_cfg_dict["containers"][i]["x"] != "None":
                containers_pose.p = gymapi.Vec3(0.35 + 0.3 * self.env_cfg_dict["containers"][i]["x"], -0.3 + 0.75 * self.env_cfg_dict["containers"][i]["y"], self.default_height/2)
            else:
                while not random_sample:
                    containers_pose.p = gymapi.Vec3(0.35 + (0.3) * random.random(), -0.3 + (0.65) * random.random(), self.default_height/2)
                    random_sample = True
                    for pose in self.containers_pose:
                        if calculate_dist(pose, containers_pose) < self.min_container_dist:
                            random_sample = False
                            break
            self.containers_pose.append(containers_pose)
            
            c = None
            if self.env_cfg_dict["containers"][i]["container_color"] == "None":
                c = "white"
            else:
                c = self.env_cfg_dict["containers"][i]["container_color"]
            rgba = to_rgba(c)
            color = gymapi.Vec3(rgba[0], rgba[1], rgba[2])
            self.containers_color.append(color)
                

    def create_butter(self):
        file_name = 'food/butter.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.butter_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        self.butter_poses = []
        butter_pose = gymapi.Transform()
        butter_pose.r = gymapi.Quat(1, 0, 0, 1)
        butter_pose.p = gymapi.Vec3(0.4, 0., self.default_height/2)
        self.butter_poses.append(butter_pose)
        butter_pose.p = gymapi.Vec3(0.4, 0.05, self.default_height/2)
        self.butter_poses.append(butter_pose)
        
    def add_butter(self, env_ptr):
        for butter_pose in self.butter_poses:
            self.butter_handle = self.gym.create_actor(env_ptr, self.butter_asset, butter_pose, "food", 0, 0)
            butter_idx = self.gym.get_actor_index(env_ptr, self.butter_handle, gymapi.DOMAIN_SIM)
            self.butter_indices.append(butter_idx)       
          
    def create_holder(self):
        file_name = 'holder/holder.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.holder_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        self.holder_pose = gymapi.Transform()
        self.holder_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.holder_pose.p = gymapi.Vec3(0.6, 0, self.default_height/2)
        
    def create_tool(self):
        
        # Load spoon asset
        self.tool_asset = {}
        self.tool_pose = {}
        self.tool_handle = {}
        self.tool_indices = {}
        tool_x = 0.41
        tool_x_between = 0.088
        tool_y = -0.365
        tool_z = self.default_height / 2
        height_offset = 0.19
        for i, tool in enumerate(self.tool_list):
            file_name = f'grab_tool/{tool}.urdf'
            self.tool_handle[tool] = []
            self.tool_indices[tool] = []
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.vhacd_enabled = True
            asset_options.fix_base_link = False
            asset_options.vhacd_params.resolution = 300000
            self.tool_asset[tool] = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
            pose = gymapi.Transform()
            pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 1.57)
            pose.p = gymapi.Vec3(tool_x,  tool_y, tool_z + height_offset)
            tool_x += tool_x_between
            self.tool_pose[tool] = pose

    def add_tool(self, env_ptr):
        for i, tool in enumerate(self.tool_list):
            handle = None
            try:
                handle = self.gym.create_actor(env_ptr, self.tool_asset[tool], self.tool_pose[tool], tool, 0, 0)
                if tool == 'spoon':
                    self.gym.set_actor_scale(env_ptr, handle, 0.75)
                cube_idx = self.gym.find_actor_rigid_body_index(env_ptr, handle, f"cube_{tool}", gymapi.DOMAIN_SIM)
                tool_idx = self.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
                self.tool_handle[tool].append(handle)
                # self.tool_indices[tool].append(tool_idx)
                self.tool_indices[tool].append(cube_idx)
                
                
                body_shape_prop = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
                # for prop in body_shape_prop:
                #     print(prop)
                # print(type(body_shape_prop[0]))
                body_shape_prop[0].friction = 10 # float("inf")
                self.gym.set_actor_rigid_shape_properties(env_ptr, handle, body_shape_prop)
            except:
                print(f'No {tool} tool')
                
    def create_ball(self):
        self.ball_radius, self.ball_mass, self.ball_friction = 0.0035, 1e-3 , 5e-3
        self.between_ball_space = 0.045
        ballGenerator = BallGenerator()
        file_name = f'BallHLS.urdf'
        ballGenerator.generate(file_name=file_name, ball_radius=self.ball_radius, ball_mass=self.ball_mass)
        self.ball_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, gymapi.AssetOptions())
    
    def set_ball_property(self, env_ptr, ball_pose, color):

        ball_friction = self.ball_friction
        ball_restitution = 0
        ball_rolling_friction = 1
        ball_torsion_friction = 1
        ball_handle = self.gym.create_actor(env_ptr, self.ball_asset, ball_pose, f"{color} grain", 0, 0)
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(env_ptr, ball_handle)

        body_shape_prop[0].friction = ball_friction
        body_shape_prop[0].rolling_friction = ball_rolling_friction
        body_shape_prop[0].torsion_friction = ball_torsion_friction
        body_shape_prop[0].restitution = ball_restitution


        self.gym.set_actor_rigid_shape_properties(env_ptr, ball_handle, body_shape_prop)
        return ball_handle
    
    def add_food(self, env_ptr):
        # add balls
        from matplotlib.colors import to_rgba
        

        ball_pose = gymapi.Transform()
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        ball_spacing = self.between_ball_space
        
        file_name = 'food/butter.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.butter_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        
        file_name = 'food/forked_food.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.forked_food_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        
        
        for i, containers_pose in enumerate(self.containers_pose):
            if self.env_cfg_dict["containers"][i]["food"] == "ball":
                total_amount = int(self.env_cfg_dict["containers"][i]["amount"])
                color_num = len(self.env_cfg_dict["containers"][i]["color"])
                ball_amount_max = min(10, total_amount)
                range = ball_spacing * 0.18 * ball_amount_max
                x_start, y_start = containers_pose.p.x - range / 2, containers_pose.p.y - range / 2
                for cnt, c in enumerate(self.env_cfg_dict["containers"][i]["color"]):
                    ball_amount = int(total_amount / color_num)
                    rgba = to_rgba(c)
                    color = gymapi.Vec3(rgba[0], rgba[1], rgba[2])
                    sub_range = ball_amount / total_amount * range
                    x, y, z = x_start, y_start, self.default_height / 2
                    while ball_amount > 0:
                        y = y_start
                        while y < y_start + sub_range and ball_amount > 0:
                            x = x_start
                            while x < x_start + range and ball_amount > 0:
                                ball_pose.p = gymapi.Vec3(x + (random.random() - 0.5) * 0.01, y + (random.random() - 0.5) * 0.01, z)
                                ball_handle = self.set_ball_property(env_ptr, ball_pose, c)
                                self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                                # self.gym.set_actor_scale(env_ptr, ball_handle, 0.5)
                                x += ball_spacing * 0.18
                                ball_amount -= 1
                            y += ball_spacing * 0.18
                        z += ball_spacing * 0.18
                    y_start += sub_range
                        
            elif self.env_cfg_dict["containers"][i]["food"] == "butter":
                butter_poses = []
                butter_pose = gymapi.Transform()
                butter_pose.r = gymapi.Quat(1, 0, 0, 1)
                butter_pose.p = gymapi.Vec3(containers_pose.p.x, containers_pose.p.y, self.default_height/2 + 0.03)
                butter_poses.append(butter_pose)
                butter_pose.p = gymapi.Vec3(containers_pose.p.x, containers_pose.p.y + 0.05, self.default_height/2 + 0.1)
                butter_poses.append(butter_pose)
                for butter_pose in butter_poses:
                    self.butter_handle = self.gym.create_actor(env_ptr, self.butter_asset, butter_pose, "cuttable_food", 0, 0)
                    butter_idx = self.gym.get_actor_index(env_ptr, self.butter_handle, gymapi.DOMAIN_SIM)
                    self.butter_indices.append(butter_idx)       
            
            elif self.env_cfg_dict["containers"][i]["food"] == "forked_food":
                pose = gymapi.Transform()
                pose.r = gymapi.Quat(1, 0, 0, 1)
                pose.p = gymapi.Vec3(containers_pose.p.x, containers_pose.p.y, self.default_height/2 + 0.03)
                self.forked_food_handle = self.gym.create_actor(env_ptr, self.forked_food_asset, pose, "forked_food", 0, 0)
                self.gym.set_actor_scale(env_ptr, self.forked_food_handle, 1.5)
                idx = self.gym.get_actor_index(env_ptr, self.forked_food_handle, gymapi.DOMAIN_SIM)
                self.forked_food_indices.append(idx)      

    def create_franka(self, reload=None):
        # create franka asset
        self.num_dofs = 0
        asset_file_franka = f"franka_description/robots/{self.env_cfg_dict['franka_type']}_franka.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.disable_gravity = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 1000000
        self.franka_asset = self.gym.load_asset(self.sim, self.asset_root, asset_file_franka, asset_options)
        self.franka_dof_names = self.gym.get_asset_dof_names(self.franka_asset)
        self.num_dofs += self.gym.get_asset_dof_count(self.franka_asset)

        self.hand_joint_index = self.gym.get_asset_joint_dict(self.franka_asset)["panda_hand_joint"]

        # # set franka dof properties
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        self.franka_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][0:7].fill(100.0)
        self.franka_dof_props["damping"][0:7].fill(40.0)
        self.franka_dof_props["stiffness"][7:9].fill(800.0)
        self.franka_dof_props["damping"][7:9].fill(40.0)

        self.franka_dof_lower_limits = self.franka_dof_props['lower']
        self.franka_dof_upper_limits = self.franka_dof_props['upper']
        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)

        # set default pose
        self.franka_start_pose = gymapi.Transform()
        self.franka_start_pose.p = gymapi.Vec3(0, 0.0, 0.2) if reload == None else gymapi.Vec3(reload[0], reload[1], reload[2])
        self.franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) if reload == None else gymapi.Quat(reload[3], reload[4], reload[5], reload[6])
    
    def add_franka(self, i, env_ptr):
        
        self.franka_indices = []
        self.franka_dof_indices = []
        self.franka_hand_indices = []
        # create franka and set properties
        franka_handle = self.gym.create_actor(env_ptr, self.franka_asset, self.franka_start_pose, "franka", i, 4, 2)
        
        
        franka_sim_index = self.gym.get_actor_index(env_ptr, franka_handle, gymapi.DOMAIN_SIM)
        self.franka_indices.append(franka_sim_index)

        franka_dof_index = [
            self.gym.find_actor_dof_index(env_ptr, franka_handle, dof_name, gymapi.DOMAIN_SIM)
            for dof_name in self.franka_dof_names
        ]
        self.franka_dof_indices.extend(franka_dof_index)

        franka_hand_sim_idx = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        self.franka_hand_indices.append(franka_hand_sim_idx)

        self.gym.set_actor_dof_properties(env_ptr, franka_handle, self.franka_dof_props)
        
        self.franka_indices = to_torch(self.franka_indices, dtype=torch.long, device=self.device)
        self.franka_dof_indices = to_torch(self.franka_dof_indices, dtype=torch.long, device=self.device)
        self.franka_hand_indices = to_torch(self.franka_hand_indices, dtype=torch.long, device=self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(spacing, 0.75 * spacing, spacing)
        self.default_height = 0.9
        self.create_table()
        self.create_container()
        self.create_ball()
        self.create_franka()
        self.create_tool()
    
        # cache some common handles for later use
        self.camera_handles = []
        self.urdf_link_indices = []
        self.containers_indices = []
        self.butter_indices = []
        self.forked_food_indices = []
        self.envs = []
        self.is_acting = {}
        self.action_stage = {}
        self.delta = {}

        # create and populate the environments
        for env_i in range(num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)
            self.add_camera(env_ptr)
            #add container
            
            for i in range(self.container_num):
                container_handle = self.gym.create_actor(env_ptr, self.container_asset[i], self.containers_pose[i], "container", 0, 0)
                self.gym.set_actor_scale(env_ptr, container_handle, 0.5)
                self.gym.set_rigid_body_color(env_ptr, container_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.containers_color[i])
                container_idx = self.gym.get_actor_rigid_body_index(env_ptr, container_handle, 0, gymapi.DOMAIN_SIM)
                self.containers_indices.append(container_idx)
            
            
            # add tabel
            self.tabel = self.gym.create_actor(env_ptr, self.table_asset, self.table_pose, "table", 0, 0)
            
            # add food
            self.add_food(env_ptr)
            # add franka
            self.add_franka(env_i, env_ptr)

            # add tool
            self.add_tool(env_ptr)


        self.containers_indices = to_torch(self.containers_indices, dtype=torch.long, device=self.device)
        self.urdf_link_indices = to_torch(self.urdf_link_indices, dtype=torch.long, device=self.device)
        self.butter_indices = to_torch(self.butter_indices, dtype=torch.long, device=self.device)
        self.forked_food_indices = to_torch(self.forked_food_indices, dtype=torch.long, device=self.device)
        for tool in self.tool_list:
            if len(self.tool_indices[tool]) > 0:
                self.tool_indices[tool] = to_torch(self.tool_indices[tool], dtype=torch.long, device=self.device) 
            else:
                self.tool_indices.pop(tool)

    def reset(self, reload=None):
        self.franka_init_pose = torch.tensor([-0.4969, -0.5425,  0.3321, -2.0888,  0.0806,  1.6983,  0.5075,  0.0400, 0.0400], dtype=torch.float32, device=self.device)
        
        if reload == None:
            self.dof_state[:, self.franka_dof_indices, 0] = self.franka_init_pose 
            self.dof_state[:, self.franka_dof_indices, 1] = 0
        else:
            dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
            print(dof_state_tensor)
            self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)
        
        target_tesnsor = self.dof_state[:, :, 0].contiguous()
        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)
        self.pos_action[:, 0:9] = target_tesnsor[:, self.franka_dof_indices[0:9]]

        franka_actor_indices = self.franka_indices.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(franka_actor_indices),
            len(franka_actor_indices)
        )

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(target_tesnsor),
            gymtorch.unwrap_tensor(franka_actor_indices),
            len(franka_actor_indices)
        )
    
        self.frame = 0
        self.action_state_reset()
        self.container_idx = 0
        self.is_moving = False
    
    def action_state_reset(self):
        self.action_list = ['scoop', 'stir', 'fork', 'cut', 'scoop_put', 'take_tool', 'put_tool']
        for action in self.action_list:
            self.action_stage[action] = -1
            self.is_acting[action] = False
            self.delta[action] = None
        self.goal_pos_set = None
        self.goal_rot_set = None
        self.goal_offset = 0.01
        self.w_offset = 0.03
        self.axis_offset = 0.03   

    def control_ik(self, dpose, damping=0.05):
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
    
    def quat_axis(self, q, axis=0):
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1
        return quat_rotate(q, basis_vec)
    
    def move(self, pos, rot=None, slow=False):
        zero = torch.Tensor([0, 0, 0])
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        if rot is None:
            rot = hand_rot
        xy_offset = 0.03
        if not self.is_moving:
            self.is_moving = True
        to_goal = pos - hand_pos
        xy_dist = torch.norm(to_goal[:, :2], dim=1)
        goal_dist = torch.norm(to_goal, dim=1)
        to_axis = rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1)
        w_dist = abs(rot[:, -1] - hand_rot[:, -1])
        pos_err = torch.where(goal_dist > self.goal_offset, pos - hand_pos, zero)
        pos_err = torch.where(xy_dist > xy_offset, torch.cat([pos_err[:, :2], torch.Tensor([[0]])], -1), pos_err)
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(rot, hand_rot), zero)
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.is_moving = False
            print("finish moving")
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        if slow:
            dpose *= 0.3      
        return dpose
    
    def take_tool(self, tool=None):
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        gripper_open = self.franka_dof_upper_limits[7:]
        gripper_close = self.franka_dof_lower_limits[7:]
        assert(tool in self.tool_indices.keys())
        if self.action_stage['take_tool'] == -1:
            # tool_pos = self.rb_state_tensor[self.tool_indices[tool], :3]
            tool_pos = self.rb_state_tensor[self.tool_indices[tool], :3]
            tool_rot = self.rb_state_tensor[self.tool_indices[tool], 3:7]
            rot = gymapi.Quat(tool_rot[:, 0], tool_rot[:, 1], tool_rot[:, 2], tool_rot[:, 3])
            roll, pitch, yaw = quaternion_to_euler(rot)
            print(f'r:{roll}, p:{pitch}, y:{yaw}')
            roll += 3.14
            rot = euler_to_quaternion(roll, pitch, yaw)
            pitch = -1.57
            knife_rot = euler_to_quaternion(roll, pitch, yaw)
            self.goal_pos_set = [
                torch.Tensor([[tool_pos[:, 0], tool_pos[:, 1] - 0.0026, tool_pos[:, 2] + 0.17]]),
                torch.Tensor([[tool_pos[:, 0], tool_pos[:, 1] - 0.0026, tool_pos[:, 2] + 0.1]]),
                torch.Tensor([[tool_pos[:, 0], tool_pos[:, 1] - 0.0026, tool_pos[:, 2] + 0.13]]),
                torch.Tensor([[tool_pos[:, 0], tool_pos[:, 1] + 0.15, tool_pos[:, 2] + 0.13]]),
                torch.Tensor([[tool_pos[:, 0], tool_pos[:, 1] + 0.15, tool_pos[:, 2] + 0.2]])
            ]
            self.goal_rot_set = [torch.Tensor([[rot.x, rot.y, rot.z, rot.w]])] * len(self.goal_pos_set)
            if tool == 'knife':
                self.gaol_rot_set[-1] = torch.Tensor([[knife_rot.x, knife_rot.y, knife_rot.z, knife_rot.w]])
            roll, pitch, yaw = quaternion_to_euler(rot)
            print(f'r:{roll}, p:{pitch}, y:{yaw}')
            self.action_stage['take_tool'] = 0
            self.is_acting['take_tool'] = True
            print(f"tool pose: {tool_pos} / {tool_rot}, goal pose: {self.goal_pos_set} / {self.goal_rot_set}")
        elif self.action_stage['take_tool'] == len(self.goal_pos_set):
            # final stage
            if(self.is_acting['take_tool']):
                print("finish taking")
                self.is_acting['take_tool'] = False
                self.franka_dof_props["effort"][7] = 0
                self.franka_dof_props["effort"][8] = 0
            return torch.Tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        
        goal_pos = self.goal_pos_set[self.action_stage['take_tool']]
        goal_rot = self.goal_rot_set[self.action_stage['take_tool']]
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > self.goal_offset, goal_pos - hand_pos, torch.Tensor([0, 0, 0]))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.Tensor([0, 0, 0]))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        self.pos_action[:, 7:9] = gripper_open
        if self.action_stage['take_tool'] > 1:
            self.pos_action[:, 7:9] = gripper_close
        else:
            self.pos_action[:, 7:9] = gripper_open
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['take_tool'] += 1
            print(self.action_stage["take_tool"])
        
        return dpose * 0.8
        
    def put_tool(self, tool: None):
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        gripper_open = self.franka_dof_upper_limits[7:]
        gripper_close = self.franka_dof_lower_limits[7:]
        assert(tool in self.tool_indices.keys())
        
        if self.action_stage['put_tool'] == -1:
            tool_pos = self.tool_pose[tool].p
            tool_rot = self.tool_pose[tool].r
            rot = tool_rot
            roll, pitch, yaw = quaternion_to_euler(rot)
            print(f'r:{roll}, p:{pitch}, y:{yaw}')
            roll += 3.14
            rot = euler_to_quaternion(roll, pitch, yaw)
            self.goal_pos_set = [
                torch.Tensor([[tool_pos.x, tool_pos.y + 0.15, tool_pos.z + 0.1]]),
                torch.Tensor([[tool_pos.x, tool_pos.y - 0.0026, tool_pos.z + 0.1]]),
                torch.Tensor([[tool_pos.x, tool_pos.y - 0.0026, tool_pos.z + 0.08]])
            ]
            self.goal_rot_set = [torch.Tensor([[rot.x, rot.y, rot.z, rot.w]])] * len(self.goal_pos_set)
            roll, pitch, yaw = quaternion_to_euler(rot)
            print(f'r:{roll}, p:{pitch}, y:{yaw}')
            self.action_stage['put_tool'] = 0
            self.is_acting['put_tool'] = True
            print(f"tool pose: {tool_pos} / {tool_rot}, goal pose: {self.goal_pos_set} / {self.goal_rot_set}")
        elif self.action_stage['put_tool'] == len(self.goal_pos_set):
            # final stage
            if(self.is_acting['put_tool']):
                print("finish taking")
                self.is_acting['put_tool'] = False
                self.franka_dof_props["effort"][7] = 0
                self.franka_dof_props["effort"][8] = 0
                self.pos_action[:, 7:9] = gripper_open
            return torch.Tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        
        goal_pos = self.goal_pos_set[self.action_stage['put_tool']]
        goal_rot = self.goal_rot_set[self.action_stage['put_tool']]
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > self.goal_offset, goal_pos - hand_pos, torch.Tensor([0, 0, 0]))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.Tensor([0, 0, 0]))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['put_tool'] += 1
            print(self.action_stage["put_tool"])
        
        return dpose * 0.8
        
        
        
        
    def scoop(self):
        
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        use_container_pos = True
        
        if self.action_stage['scoop'] == -1:
            # initialize
            
            print("scoop start")
            self.delta['scoop'] = [0.5, 0.5, 0.5, 0.3]
            # self.delta['scoop'] = [0.05, 0.05, 0.05, 0.5]
            self.goal_pos_set = [hand_pos + torch.Tensor([[0., 0., -0.065]])]
            self.goal_rot_set = [torch.Tensor([[1.0, 0.0, -0.05, 0.0]])]
            init_pos = hand_pos.clone()
            init_pos[:, 2] = 0
            
            # find the nearest container
            if use_container_pos:
                min_dist = float('inf')
                for i, container_pose in enumerate(self.containers_pose):
                    container_p = torch.Tensor([[container_pose.p.x, container_pose.p.y, 0.0]])
                    print(f"container pos: {container_p}, init pos: {init_pos}")
                    dist = torch.norm(container_p - init_pos, dim=1)
                    if dist < min_dist:
                        min_dist = dist
                        best_tensor = container_p
                init_pos = best_tensor - torch.Tensor([[0.01, -0.02, 0.0]])
                init_pos[0][2] = 0.03
                init_pos[0][1] += 0.0127
                #init_pos[0][0] -= 0.02
            else:
                init_pos[0][2] -= 0.02
                
            x_temp = -0.08
            
            # original pos: 0.3871, 0.0877, container pos: 0.43999999999999995, 0.07500000000000001
            self.goal_pos_set = [
                init_pos + torch.tensor([[0., 0.0000, 0.7836]]), 
                init_pos + torch.tensor([[-0.03,  0.0000,  0.7228]]), 
                init_pos + torch.tensor([[-0.0299,  0.0109,  0.6728]]), 
                init_pos + torch.tensor([[-0.0271,  0.0050,  0.6808]]), 
                init_pos + torch.tensor([[-0.0264,  0.0054,  0.6716]]), 
                init_pos + torch.tensor([[0.0169, 0.0043, 0.6700]]), 
                init_pos + torch.tensor([[x_temp+0.0512, 0.0043, 0.6638]]), 
                init_pos + torch.tensor([[x_temp+0.0505, 0.0043, 0.6638]]),
                init_pos + torch.tensor([[x_temp+0.0513-0.0013, 0.0038, 0.6650]]), # 8
                init_pos + torch.tensor([[x_temp+0.0448, 0.0035, 0.6668]]), 
                init_pos + torch.tensor([[x_temp+0.0295, 0.0031, 0.6698]]), 
                init_pos + torch.tensor([[x_temp+-0.0007,  0.0022,  0.6728]]), 
                init_pos + torch.tensor([[x_temp+-0.0224,  0.0019,  0.6758]]), 
                init_pos + torch.tensor([[x_temp+-0.0216,  0.0021,  0.6708]]), 
                init_pos + torch.tensor([[x_temp+-0.0570,  0.0019,  0.6808]]), 
                init_pos + torch.tensor([[x_temp+-0.1330,  0.0017,  0.7000]])
            ]
            self.goal_rot_set = [
                torch.tensor([[ 0.9945,  0.0413, -0.0809,  0.0523]]),
                torch.tensor([[ 0.9945,  0.0410, -0.0808,  0.0522]]),
                torch.tensor([[ 0.9953,  0.0393, -0.0701,  0.0542]]),
                torch.tensor([[ 0.9976,  0.0345, -0.0243,  0.0549]]),
                torch.tensor([[ 0.9977,  0.0345, -0.0176,  0.0563]]),
                torch.tensor([[0.9975, 0.0310, 0, 0.0575]]),
                torch.tensor([[0.9975, 0.0310, 0.0288, 0.0575]]),
                torch.tensor([[0.9965, 0.0388, 0.0500, 0.0580]]),
                torch.tensor([[0.9952, 0.0278, 0.0736, 0.0586]]), # 8
                torch.tensor([[0.9925, 0.0257, 0.1034, 0.0594]]),
                torch.tensor([[0.9869, 0.0225, 0.1478, 0.0604]]),
                torch.tensor([[0.9621, 0.0144, 0.2648, 0.0626]]),
                torch.tensor([[0.9340, 0.0089, 0.3514, 0.0637]]),
                torch.tensor([[0.9337, 0.0093, 0.4071, 0.0639]]),
                torch.tensor([[0.8848, 0.0023, 0.5116, 0.0641]]),
                torch.tensor([[0.8844, 0.0025, 0.5116, 0.0640]])
            ]
            self.action_stage['scoop'] = 0
            self.is_acting['scoop'] = True
            return torch.Tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        
        elif self.action_stage['scoop'] == len(self.goal_pos_set):
            # final stage
            if self.is_acting['scoop']:
                print("finish scoop")
                self.is_acting['scoop'] = False
            return torch.Tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        
        goal_pos = self.goal_pos_set[self.action_stage['scoop']]
        goal_rot = self.goal_rot_set[self.action_stage['scoop']]
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > self.goal_offset, goal_pos - hand_pos, torch.Tensor([0, 0, 0]))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.Tensor([0, 0, 0]))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['scoop'] += 1
            print(self.action_stage['scoop'])
        dpose *= self.delta['scoop'][self.action_stage['scoop'] if self.action_stage['scoop'] < len(self.delta['scoop']) else -1]
        
        return dpose
    
    def scoop_put(self):
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        delta = 1.5
        
        if self.action_stage['scoop_put'] == -1:
            print("scoop put")
            goal_rot = torch.Tensor([[1, 0, -0.1, 0]])
            stage_num = 4
            goal_rot = (goal_rot - hand_rot) / stage_num
            self.goal_rot_set = [
                hand_rot + goal_rot * (i + 1) for i in range(stage_num)
            ]
            self.action_stage['scoop_put'] = 0
            self.is_acting['scoop_put'] = True
            return torch.Tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        elif self.action_stage['scoop_put'] == len(self.goal_rot_set):
            # final stage
            print("finish putting")
            self.is_acting['scoop_put'] = False
            return torch.Tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        
        goal_rot = self.goal_rot_set[self.action_stage['scoop_put']]
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.Tensor([[0, 0, 0]])
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.Tensor([0, 0, 0]))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        if axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['scoop_put'] += 1
        dpose *= delta
        return dpose
        
    def stir(self):
        
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        
        
        if self.action_stage['stir'] == -1:
            self.stir_center = hand_pos + torch.Tensor([[0.03, 0, 0]])
            self.goal_rot_set = [
                self.stir_center + torch.Tensor([[0.1, 0, 0]]),
                hand_pos
            ]
            self.action_stage['stir'] = 0
            self.is_acting['stir'] = True
            self.delta['stir'] = 0.1
            return torch.Tensor([[0.], [0.], [-5.], [0.], [0.], [0.]])
            
        elif self.action_stage['stir'] == len(self.goal_rot_set):
            print("finish stirring")
            self.is_acting['stir'] = False
            return torch.Tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        
        stage_xy = self.goal_rot_set[self.action_stage['stir']]
        stage_dir = stage_xy - hand_pos
        stage_dist = torch.norm(stage_dir[:, :1], dim=1).unsqueeze(-1)
        if stage_dist < self.goal_offset:
            self.action_stage['stir'] += 1
        
        container_dir = self.stir_center - hand_pos
        container_dist = torch.norm(container_dir[:, :1], dim=1).unsqueeze(-1)
        stir_radius = 1
        if container_dist > stir_radius:
            print("reset", container_dir, container_dist)
            self.is_acting['stir'] = False
            return torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
        goal_xy = container_dir[:, [1, 0, 2]] * torch.Tensor([[-1, 1, 0]])
        goal_norm = torch.norm(goal_xy, dim=1).unsqueeze(-1)
        goal_xy /= goal_norm
        pos_err = goal_xy
        orn_err = torch.Tensor([[0, 0, 0]])
        
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)        
        return dpose
        
    def fork(self):
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        food_pos = self.rb_state_tensor[self.forked_food_indices, :3]
        food_rot = self.rb_state_tensor[self.forked_food_indices, 3:7]
        self.delta['fork'] = 0.3
        if self.action_stage['fork'] == -1:
            self.is_acting['fork'] = True
            self.action_stage['fork'] = 0
        elif self.action_stage['fork'] == 2:
            print("finish forking")
            self.is_acting['fork'] = False
            return torch.Tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        fork_offset = 3e-3
        goal_pos = food_pos
        goal_pos[:, 2] += 0.4 if self.action_stage['fork'] == 0 else 0.2
        goal_pos[:, 1] -= 0.023
        # goal_pos[:, 0] -= 0.002
        roll, pitch, yaw = quaternion_to_euler(gymapi.Quat(food_rot[:, 0], food_rot[:, 1], food_rot[:, 2], food_rot[:, 3]))
        roll += 1.57
        yaw += 1.57
        goal_rot = euler_to_quaternion(roll, pitch, yaw)
        goal_rot = torch.Tensor([[goal_rot.x, goal_rot.y, goal_rot.z, goal_rot.w]])
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal[..., :2], dim=1).unsqueeze(-1) if self.action_stage['fork'] == 0 else torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > fork_offset, goal_pos - hand_pos, torch.Tensor([0, 0, 0]))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.Tensor([0, 0, 0]))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        if goal_dist <= fork_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['fork'] += 1
        dpose *= self.delta['fork']
        return dpose
        
    
    def cut(self):
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        
        cut_delta = 0.1
        
        if self.action_stage['cut'] == -1:
            self.action_stage['cut'] = 0
            self.is_acting['cut'] = True
            self.cut_end = self.default_height / 2 + 0.12
            return torch.Tensor([[[0.], [0.], [0.], [0.], [0.], [0.]]])
        if hand_pos[:, -1].item() - self.cut_end < 0.001:
            print("finish cut")
            self.action_stage['cut'] = -1
            self.is_acting['cut'] = False
            return torch.Tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        return torch.Tensor([[[0.], [0.], [self.cut_end - hand_pos[:, -1].item()], [0.], [0.], [0.]]]) * cut_delta
        
    def switch(self):
        next_tool = np.random.choice(self.tool_list, 1)[0]
        for i in range(len(self.tool_list)):
            if self.tool_list[i] == self.tool:
                next_tool = self.tool_list[(i + 1) % len(self.tool_list)]
                break
        print(next_tool)
        self.tool = next_tool
        reload_start_pose = self.rb_state_tensor[self.franka_hand_indices, :]
        reload_init_pose = self.dof_state[:, self.franka_dof_indices, 0]
        reload_start_pose = [x.item() for x in reload_start_pose.squeeze(-1)[0]]
        reload_init_pose = reload_init_pose.squeeze(-1)[0]
        print(reload_init_pose)
        self.create_franka(reload_start_pose)
        for i in range(self.num_envs):
            self.add_franka(i, self.envs[i])
        self.reset(reload_init_pose)
        
    def data_collection(self):
        self.reset()
        img_num = 0

        action = ""

        while not self.gym.query_viewer_has_closed(self.viewer):
            if self.container_num > 0:
                container_pos = [self.rb_state_tensor[indice, :3] for indice in self.containers_indices]
                scoop_idx = 1
                put_idx = -1
                scoop_pos = container_pos[scoop_idx] + torch.Tensor([[-0.1, 0., 0.3]])
                scoop_rot = torch.Tensor([[1., 0., 0., 0.]])
                scoop_put_pos = container_pos[put_idx] + torch.Tensor([[-0.15, 0., 0.3]])
                stir_pos = container_pos[scoop_idx] + torch.Tensor([[-0.06, 0., 0.3]])
                stir_rot = torch.Tensor([[1., 0., 0., 0.]])
                fork_food_pos = self.rb_state_tensor[self.butter_indices, :3]
                fork_food_pos[:, -1] += 0.3
                fork_food_rot = torch.Tensor([[1., 0., 0., 0.]])
            # scoop_rot = quat_from_angle_axis(torch.Tensor([0.3]), to_container)
            # scoop_rot = torch.cat([to_container.view(3, ), torch.Tensor([0.8])], -1).unsqueeze(-1).view(1, 4)
            
            # ciritical_yaw_dir = self.quat_axis(scoop_rot, 0)
            # hand_yaw_dir = self.quat_axis(hand_rot, 0)
            # yaw_dot = torch.bmm(ciritical_yaw_dir.view(self.num_envs, 1, 3), hand_yaw_dir.view(self.num_envs, 3, 1)).squeeze(-1)
            
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.render_all_camera_sensors(self.sim)

            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)

            gripper_open = self.franka_dof_upper_limits[7:]
            gripper_close = self.franka_dof_lower_limits[7:]
            delta = 0.05
            moving_delta = 0.5
            print_pos = True
            
            #print(self.rb_state_tensor[self.franka_hand_indices,:])

            
            for evt in self.gym.query_viewer_action_events(self.viewer):
                action = evt.action if (evt.value) > 0 else ""
            
            for a in self.action_list:
                if self.is_acting[a]:
                    if action in [a, ""]:
                        action = a
                    elif action != "save":
                        self.action_state_reset()
                        print(f"{a} terminate")
                    break
                
            if action == "up":
                dpose = torch.Tensor([[[0.],[0.],[1.],[0.],[0.],[0.]]]) * delta
            elif action == "down":
                dpose = torch.Tensor([[[0.],[0.],[-1.],[0.],[0.],[0.]]]) * delta
            elif action == "left":
                dpose = torch.Tensor([[[0.],[-1.],[0.],[0.],[0.],[0.]]]) * delta
            elif action == "right":
                dpose = torch.Tensor([[[0.],[1.],[0.],[0.],[0.],[0.]]]) * delta
            elif action == "backward":
                dpose = torch.Tensor([[[-1.],[0.],[0.],[0.],[0.],[0.]]]) * delta
            elif action == "forward":
                dpose = torch.Tensor([[[1.],[0.],[0.],[0.],[0.],[0.]]]) * delta
            elif action == "turn_left":
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[-10.]]]) * delta
            elif action == "turn_right":
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[10.]]]) * delta
            elif action == "turn_up":
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[10.],[0.]]]) * delta
            elif action == "turn_down":
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[-10.],[0.]]]) * delta
            elif action == "test_pos":
                dpose = torch.Tensor([[[0.],[0.],[0.],[1.],[0.],[0.]]]) * delta
            elif action == "test_neg":
                dpose = torch.Tensor([[[0.],[0.],[0.],[-1.],[0.],[0.]]]) * delta
            elif action == "scoop":
                dpose = self.scoop()
            elif action == "stir":
                dpose = self.stir() * self.delta['stir']
            elif action == "scoop_put":
                dpose = self.scoop_put()
            elif action == "move_stir":
                dpose = self.move(stir_pos, stir_rot) * moving_delta
            elif action == "move_scoop":
                dpose = self.move(scoop_pos, scoop_rot) * moving_delta
            elif action == "move_scoop_put":
                dpose = self.move(scoop_put_pos, slow=True) * moving_delta
            elif action == "move_fork":
                dpose = self.move(fork_food_pos, fork_food_rot) * moving_delta
            elif action == "fork":
                dpose = self.fork()
            elif action == "cut":
                dpose = self.cut()
            elif action == "take_tool":
                dpose = self.take_tool("spoon")
            elif action == "put_tool":
                dpose = self.put_tool("spoon")
            elif action == "change container index":
                self.container_idx = int(input("New goal container index"))
                if self.container_idx >= self.container_num: self.container_idx = self.container_num - 1
                elif self.container_idx < 0: self.container_idx = 0
            elif action == "switch":
                self.switch()
            elif action == "gripper_close":
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
                if torch.all(self.pos_action[:, 7:9] == gripper_close):
                    self.pos_action[:, 7:9] = gripper_open
                elif torch.all(self.pos_action[:, 7:9] == gripper_open):
                    self.pos_action[:, 7:9] = gripper_close
            elif action == "save":
                hand_pos = self.rb_state_tensor[self.franka_hand_indices, 0:3]
                hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
                print(hand_pos)
                print(hand_rot)
                print(self.pos_action[:, 7:9])
    
                self.record.append([hand_pos, hand_rot])
                for i in range(self.num_envs):
                    file_name = f'./observation/{self.output_folder}/image{img_num}.png'
                    img_num += 1
                    self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR, file_name)
            elif action == "to_file":
                pose = [i[0].numpy().squeeze(0) for i in self.record]
                rot = [i[1].numpy().squeeze(0) for i in self.record]
                first = pose[0]
                first[2] = 0
                pose = [oldpos - first for oldpos in pose]
                with open('./pose.txt', 'w') as f:
                    for row in pose:
                        f.write("[" + ", ".join(map(str, row)) + "]" + '\n')
                with open('./rot.txt', 'w') as f:
                    for row in rot:
                        f.write("[" + ", ".join(map(str, row)) + "]" + '\n')
    
            elif action == "quit":
                break
            else:
                print_pos = False
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
            dpose = dpose.to(self.device)
            
            
            self.pos_action[:, :7] = self.dof_state[:, self.franka_dof_indices, 0].squeeze(-1)[:, :7] + self.control_ik(dpose)
       
            test_dof_state = self.dof_state[:, :, 0].contiguous()
            test_dof_state[:, self.franka_dof_indices] = self.pos_action

            franka_actor_indices = self.franka_indices.to(dtype=torch.int32)
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(test_dof_state),
                gymtorch.unwrap_tensor(franka_actor_indices),
                len(franka_actor_indices)
            )

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

            self.frame += 1

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def simulate(self):
        self.reset()

        grasp_pos = torch.tensor([[ 0, 0,  0]], dtype=torch.float32, device=self.device) # [ 0.5064, -0.1349,  0.4970]
        grasp_rot = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32, device=self.device)

        stage = 0

        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.render_all_camera_sensors(self.sim)

            # refresh tensor
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)

            hand_pos = self.rb_state_tensor[self.franka_hand_indices, 0:3]
            hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]

            # compute open pose
            grasp_matrix = pose7d_to_matrix(torch.cat([grasp_pos, grasp_rot], dim=1))
            urdf_matrix = pose7d_to_matrix(self.rb_state_tensor[self.urdf_link_indices, 0:7])

            rotate_m90_matrix = torch.eye(4, dtype=torch.float32, device=self.device).reshape(1, 4, 4).repeat(self.num_envs, 1, 1)
            rotate_m90_matrix[:, :3, :3] = pytorch3d.transforms.euler_angles_to_matrix(
                torch.tensor([[0, -torch.pi / 2, 0]], dtype=torch.float32, device=self.device), "XYZ"
            )
            
            open_matrix = urdf_matrix @ rotate_m90_matrix @ torch.linalg.inv(urdf_matrix) @ grasp_matrix
            open_pose_7d = matrix_to_pose_7d(open_matrix)

            open_pos = open_pose_7d[:, 0:3]
            open_rot = open_pose_7d[:, 3:7]

            goal_pos_list = [hand_pos, grasp_pos, grasp_pos, open_pos]
            goal_rot_list = [hand_rot, grasp_rot, grasp_rot, open_rot]

            # check stage changes
            print(self.frame)
            if self.frame > 400:
                stage = 1

            if self.frame > 600:
                stage = 2
                self.pos_action[:, 7:9] = 0
            
            if self.frame > 800:
                stage = 3

            # ik control
            pos_err = goal_pos_list[stage] - hand_pos
            orn_err = self.orientation_error(goal_rot_list[stage], hand_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

            self.pos_action[:, :7] = self.dof_state[:, self.franka_dof_indices, 0].squeeze(-1)[:, :7] + self.control_ik(dpose)

            test_dof_state = self.dof_state[:, :, 0].contiguous()
            test_dof_state[:, self.franka_dof_indices] = self.pos_action

            franka_actor_indices = self.franka_indices.to(dtype=torch.int32)
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(test_dof_state),
                gymtorch.unwrap_tensor(franka_actor_indices),
                len(franka_actor_indices)
            )

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

            self.frame += 1
            

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

def read_yaml(file_path, env_type='simple_env', env_num=1):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    config = config[env_type][env_num]
    return config



if __name__ == "__main__":
    config = read_yaml("config.yaml")
    issac = IsaacSim(config)
    issac.data_collection()

    issac.simulate()
