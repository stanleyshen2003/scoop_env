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
from isaacgym.torch_utils import to_torch, quat_rotate, quat_mul, quat_conjugate

import os
import time
import random
import yaml
import torch
import numpy as np
import pytorch3d.transforms
import math
import sapien
from time import time, sleep
from PIL import Image
from typing import List
torch.pi = math.pi

from src import Decision_pipeline, BallGenerator
from src import action_state


def pose7d_to_matrix(pose7d: torch.tensor):
    matrix = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).repeat(pose7d.shape[0], 1, 1)
    matrix[:, :3, :3] = pytorch3d.transforms.quaternion_to_matrix(pose7d[:, [6, 3, 4, 5]])
    matrix[:, :3, 3] = pose7d[:, :3]

    return matrix

def matrix_to_pose_7d(matrix: torch.tensor):
    pose_7d = torch.zeros((matrix.shape[0], 7), dtype=torch.float32)
    pose_7d[:, 3:] = pytorch3d.transforms.matrix_to_quaternion(matrix[:, :3, :3])[:, [1, 2, 3, 0]]
    pose_7d[:, :3] = matrix[:, :3, 3]

    return pose_7d

def euler_to_quaternion(roll, pitch, yaw, use_tensor=False):
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

    return q if not use_tensor else torch.tensor([[q.x, q.y, q.z, q.w]])

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
        keyboard_modes = ['collect', 'demo']
        custom_param = [
            {'name': '--output', 'type': str, 'default': None, 'help': 'Output folder'},
            {'name': '--k_mode', 'type': str, 'default': 'collect', 'help': f"Supported mode [{'|'.join(keyboard_modes)}]"},
            {'name': '--use_gpu', 'action': 'store_true'}
        ]
        args = gymutil.parse_arguments(description="Joint control Methods Example", custom_parameters=custom_param)
        self.device = 'cuda:0' if args.use_gpu and torch.cuda.is_available() else "cpu"
        
        self.env_cfg_dict = env_cfg_dict
        self.instruction =  self.env_cfg_dict["instruction"] if self.env_cfg_dict["instruction"] != "None" \
            else "Stir the beans in the bowl, then scoop it to the round plate."
        self.tool_list = np.array(["spoon", "fork", "knife"]) if self.env_cfg_dict["tool"] == "None" else self.env_cfg_dict["tool"]
        
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # create simulator
        self.env_spacing = 1.5
        self.max_episode_length = 195
        self.asset_root = "urdf"
        self.gravity = -1
        self.use_container_pos = True
        
        # if there is already one sim, close it
        self.create_sim(args)
        
        # create viewer using the default camera properties
        viewer_props = gymapi.CameraProperties()
        viewer_props.width = 1080
        viewer_props.height = 720
        self.viewer = self.gym.create_viewer(self.sim, viewer_props)

        # keyboard event
        k_mode = args.k_mode
        assert k_mode in keyboard_modes, f"Unsupported mode: {k_mode} "
        print(f"Keyboard mode: {k_mode}")
        
        if k_mode == 'collect':
            setting = {
                "UP": "up",
                "DOWN": "down",
                "LEFT": "left",
                "RIGHT": "right",
                "W": "backward",
                "S": "forward",
                "A": "turn_right",
                "D": "turn_left",
                "E": "turn_up",
                "Q": "turn_down",
                "SPACE": "gripper_close",
                "C": "choose action",
                "T": "pull_bowl_closer", 
            }
        elif k_mode == 'demo':
            setting = {
                "0": "set_tool",
                "Q": "scoop",
                "W": "scoop_put",
                "E": "stir",
                "A": "fork",
                "S": "cut",
                "T": "take_tool",
                "Y": "put_tool",
                "C": "move_around",
                "D": "pull_bowl_closer",
            }
            for i, container in enumerate(self.containers_indices):
                if i + 1 > 9:
                    break
                setting[str(i + 1)] = f"move_{container}"
            self.action_list = list(setting.values())
            
        setting['X'] = 'save'
        setting['F'] = 'to_file'
        setting['ESCAPE'] = 'quit'
        self.set_keyboard(setting)
        
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
        
        self.decision_pipeline = Decision_pipeline(self.containers_list)
        

    def set_keyboard(self, setting: dict):
        for key_name, description in setting.items():
            print(f"{key_name: >7} : {description}")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, getattr(gymapi, f"KEY_{key_name}"), description)
    
    def create_sim(self, args):
        # parse arguments
        
        args.use_gpu_pipeline = args.use_gpu
        self.num_envs = 1
        self.action_sequence = []
        
        self.output_folder = args.output
        
        if not self.output_folder is None and not os.path.exists(f"./observation/{self.output_folder}"):
            os.mkdir(f"./observation/{self.output_folder}")

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, self.gravity)

        #黏稠度
        sim_params.dt = 1.0 / 70

        sim_params.substeps = 1

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1

        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.contact_offset = 0.0001
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.max_depenetration_velocity = 1000

        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        
        
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))
        self.gym.prepare_sim(self.sim)
        
    def add_camera(self, env_ptr):
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1920
        camera_props.height = 1080
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
        asset_options.fix_base_link = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.table_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        # self.table_dims = gymapi.Vec3(0.8, 1, self.default_height)
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
        asset_options.fix_base_link = False
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
        self.containers_indices = {}
        self.containers_pose = []
        self.containers_color = []
        self.containers_list = []
        self.min_container_dist = 0.2
        max_x = 0.58
        min_x = 0.35
        max_y = 0.34
        min_y = -0.2
        for i in range(self.container_num):
            random_sample = False
            container_type = self.env_cfg_dict["containers"][i]["type"]
            
            container_pose = gymapi.Transform()
            container_pose.r = gymapi.Quat(1, 0, 0, 1) if 'plate' not in container_type else gymapi.Quat(0, 0, 0, 1)
            height_offset = 0.02 if container_type == "bowl" else 0
            if self.env_cfg_dict["containers"][i]["x"] != "None":
                container_pose.p = gymapi.Vec3(self.env_cfg_dict["containers"][i]["x"], self.env_cfg_dict["containers"][i]["y"], self.default_height / 2 + height_offset)
            else:
                while not random_sample:
                    container_pose.p = gymapi.Vec3(min_x + (max_x - min_x) * random.random(), min_y + (max_y - min_y) * random.random(), self.default_height / 2 + height_offset)
                    random_sample = True
                    for pose in self.containers_pose:
                        if calculate_dist(pose, container_pose) < self.min_container_dist:
                            random_sample = False
                            break
            
            c = None
            if self.env_cfg_dict["containers"][i]["container_color"] == None:
                c = "white"
            else:
                c = self.env_cfg_dict["containers"][i]["container_color"]
            
            food = None
            if self.env_cfg_dict["containers"][i]["food"] == "None":
                food = "(empty)"
            elif self.env_cfg_dict["containers"][i]["food"] == "ball":
                food_colors = " and ".join(self.env_cfg_dict["containers"][i]["color"])
                food = f"(with {food_colors} beans)"                
            else:
                food = f"(with {self.env_cfg_dict['containers'][i]['food']})"
                
            rgba = to_rgba(c)
            color = gymapi.Vec3(rgba[0], rgba[1], rgba[2])
            self.containers_indices[f"{c}_{container_type} {food}"] = []
            self.containers_list.append(f"{c}_{container_type} {food}")
            self.containers_pose.append(container_pose)
            self.containers_color.append(color)
        print(self.containers_list)
                
    def add_container(self, env_ptr):
        for i in range(self.container_num):
            container_handle = self.gym.create_actor(env_ptr, self.container_asset[i], self.containers_pose[i], self.containers_list[i], 0, 0)
            self.gym.set_actor_scale(env_ptr, container_handle, 0.5)
            self.gym.set_rigid_body_color(env_ptr, container_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.containers_color[i])
            container_idx = self.gym.get_actor_rigid_body_index(env_ptr, container_handle, 0, gymapi.DOMAIN_SIM)
            self.containers_indices[self.containers_list[i]].append(container_idx)

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
        tool_x = 0.412
        tool_x_between = 0.086
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
        self.ball_radius, self.ball_mass, self.ball_friction = 0.0035, 1e-3, 1e-3
        self.between_ball_space = 0.045
        ballGenerator = BallGenerator()
        file_name = f'ball/BallHLS.urdf'
        ballGenerator.generate(root=self.asset_root, file_name=file_name, ball_radius=self.ball_radius, ball_mass=self.ball_mass)
        self.ball_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, gymapi.AssetOptions())
    
    def set_ball_property(self, env_ptr, ball_pose, color):

        ball_friction = self.ball_friction
        ball_restitution = 0
        ball_rolling_friction = 0.1
        ball_torsion_friction = 0.1
        ball_handle = self.gym.create_actor(env_ptr, self.ball_asset, ball_pose, f"{color} grain", 0, 0)
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(env_ptr, ball_handle)

        body_shape_prop[0].friction = ball_friction
        body_shape_prop[0].rolling_friction = ball_rolling_friction
        body_shape_prop[0].torsion_friction = ball_torsion_friction
        body_shape_prop[0].restitution = ball_restitution


        self.gym.set_actor_rigid_shape_properties(env_ptr, ball_handle, body_shape_prop)
        return ball_handle
    
    def create_food(self):
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
        
    def add_food(self, env_ptr):
        # add balls
        from matplotlib.colors import to_rgba
        

        ball_pose = gymapi.Transform()
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        ball_spacing = self.between_ball_space
        for i, containers_pose in enumerate(self.containers_pose):
            if self.env_cfg_dict["containers"][i]["food"] == "ball":
                total_amount = int(self.env_cfg_dict["containers"][i]["amount"])
                color_num = len(self.env_cfg_dict["containers"][i]["color"])
                ball_amount_max = min(10, total_amount)
                ran = ball_spacing * 0.18 * ball_amount_max
                x_start, y_start = containers_pose.p.x - ran / 2, containers_pose.p.y - ran / 2
                for cnt, c in enumerate(self.env_cfg_dict["containers"][i]["color"]):
                    ball_amount = int(total_amount / color_num)
                    rgba = to_rgba(c)
                    color = gymapi.Vec3(rgba[0], rgba[1], rgba[2])
                    sub_ran = ball_amount / total_amount * ran
                    x, y, z = x_start, y_start, self.default_height / 2 + 0.02
                    while ball_amount > 0:
                        y = y_start
                        while y < y_start + sub_ran and ball_amount > 0:
                            x = x_start
                            while x < x_start + ran and ball_amount > 0:
                                ball_pose.p = gymapi.Vec3(x + (random.random() - 0.5) * 0.01, y + (random.random() - 0.5) * 0.01, z)
                                ball_handle = self.set_ball_property(env_ptr, ball_pose, c)
                                self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                                # self.gym.set_actor_scale(env_ptr, ball_handle, 0.5)
                                x += ball_spacing * 0.18
                                ball_amount -= 1
                            y += ball_spacing * 0.18
                        z += ball_spacing * 0.18
                    y_start += sub_ran
                        
            elif self.env_cfg_dict["containers"][i]["food"] == "cuttable_food":
                rot = gymapi.Quat(1, 0, 0, 1)
                x, y, z = containers_pose.p.x, containers_pose.p.y, containers_pose.p.z + 0.03
                for _ in range(2):
                    pose = gymapi.Transform()
                    pose.r = rot
                    pose.p = gymapi.Vec3(x, y, z)
                    y += 0.05
                    self.butter_handle = self.gym.create_actor(env_ptr, self.forked_food_asset, pose, "cuttable_food", 0, 0)
                    butter_idx = self.gym.get_actor_index(env_ptr, self.butter_handle, gymapi.DOMAIN_SIM)
                    self.butter_indices.append(butter_idx)    
                    self.forked_food_indices.append(butter_idx)    
            
            elif self.env_cfg_dict["containers"][i]["food"] == "forked_food":
                pose = gymapi.Transform()
                pose.r = gymapi.Quat(1, 0, 0, 1)
                pose.p = gymapi.Vec3(containers_pose.p.x, containers_pose.p.y, self.default_height/2 + 0.03)
                self.forked_food_handle = self.gym.create_actor(env_ptr, self.forked_food_asset, pose, "forked_food", 0, 0)
                idx = self.gym.get_actor_index(env_ptr, self.forked_food_handle, gymapi.DOMAIN_SIM)
                self.forked_food_indices.append(idx)

    def create_microwave(self):
        file_name = 'microwave/mobility.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.microwave_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        self.num_dofs += self.gym.get_asset_dof_count(self.microwave_asset)
        self.microwave_dof_props = self.gym.get_asset_dof_properties(self.microwave_asset)
        self.microwave_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        
        self.microwave_pose = gymapi.Transform()
        quat = euler_to_quaternion(0, 0, math.pi / 2)
        self.microwave_pose.r = quat
        self.microwave_pose.p = gymapi.Vec3(0.5, 0.6, self.default_height / 2 + 0.095)
    
    def add_microwave(self, env_ptr):
        microwave_handle = self.gym.create_actor(env_ptr, self.microwave_asset, self.microwave_pose, 'microwave', 0, 8)
        self.gym.set_actor_scale(env_ptr, microwave_handle, 0.3)
        self.gym.set_actor_dof_properties(env_ptr, microwave_handle, self.microwave_dof_props)
        self.microwave_door_indices = self.gym.find_actor_dof_index(env_ptr, microwave_handle, 'door', gymapi.DOMAIN_SIM)
        self.microwave_door_indices = to_torch(self.microwave_door_indices, dtype=torch.long, device=self.device)
        microwave_idx = self.gym.get_actor_index(env_ptr, microwave_handle, gymapi.DOMAIN_SIM)
        self.microwave_indices.append(microwave_idx)
        
    def create_franka(self, reload=None):
        # create franka asset
        self.num_dofs = 0
        asset_file_franka = "franka_description/robots/original_franka.urdf"
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
        self.create_food()
        self.create_microwave()
    
        # cache some common handles for later use
        self.camera_handles = []
        self.urdf_link_indices = []
        self.butter_indices = []
        self.forked_food_indices = []
        self.microwave_indices = []
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
            self.add_container(env_ptr)
            self.table = self.gym.create_actor(env_ptr, self.table_asset, self.table_pose, "table", 0, 8)
            self.add_food(env_ptr)
            self.add_franka(env_i, env_ptr)
            self.add_tool(env_ptr)
            self.add_microwave(env_ptr)

        self.urdf_link_indices = to_torch(self.urdf_link_indices, dtype=torch.long, device=self.device)
        self.butter_indices = to_torch(self.butter_indices, dtype=torch.long, device=self.device)
        self.forked_food_indices = to_torch(self.forked_food_indices, dtype=torch.long, device=self.device)
        self.microwave_indices = to_torch(self.microwave_indices, dtype=torch.long, device=self.device)
        for container in self.containers_list:
            if len(self.containers_indices[container]) > 0:
                self.containers_indices[container] = to_torch(self.containers_indices[container], dtype=torch.long, device=self.device) 
            else:
                self.containers_indices.pop(container)
                
        for tool in self.tool_list:
            if len(self.tool_indices[tool]) > 0:
                self.tool_indices[tool] = to_torch(self.tool_indices[tool], dtype=torch.long, device=self.device) 
            else:
                self.tool_indices.pop(tool)

    def reset(self):
        self.franka_init_pose = torch.tensor([-0.4969, -0.5425,  0.3321, -2.0888,  0.0806,  1.6983,  0.5075,  0.0400, 0.0400], dtype=torch.float32, device=self.device)
        self.dof_state[:, self.franka_dof_indices, 0] = self.franka_init_pose 
        self.dof_state[:, self.franka_dof_indices, 1] = 0
        self.dof_state[:, self.microwave_door_indices, 0] = math.pi # math.pi / 18
        self.dof_state[:, self.microwave_door_indices, 1] = 0
            
        target_tesnsor = self.dof_state[:, :, 0].contiguous()

        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)
        self.pos_action[:, 0:9] = target_tesnsor[:, self.franka_dof_indices[0:9]]

        franka_actor_indices = self.franka_indices.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state)
        )

        self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(target_tesnsor)
        )
        self.frame = 0

        self.action_state_reset()

        self.container_idx = 0
    
    def action_state_reset(self):
        self.action_stage = {}
        self.is_acting = {}
        self.delta = {}
        self.action_init = {}
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
    
    def reach_goal_position(self, goal_dist=float("-inf"), axis_dist=float("-inf"), w_dist=float("-inf")):
        return goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset
    
    def move(self, object: str, pos=None, rot=None, slow=False):
        object_type = "tool" if object in self.tool_list else "container"
        zero = torch.tensor([0, 0, 0], device=self.device)
            
        self.indices_list = {
            "tool": self.tool_indices,
            "container": self.containers_indices
        }
        
        object_indice = self.indices_list[object_type][object]
        object_pos = self.rb_state_tensor[object_indice, :3] + torch.tensor([-0.05, 0, 0.4], device=self.device)
        
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        if rot is None:
            rot = hand_rot
        if pos is None:
            pos = object_pos
        xy_offset = 0.03
        to_goal = pos - hand_pos
        xy_dist = torch.norm(to_goal[:, :2], dim=1)
        goal_dist = torch.norm(to_goal, dim=1)
        to_axis = rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1)
        w_dist = abs(rot[:, -1] - hand_rot[:, -1])
        if not self.is_acting[f'move_{object}'] and not self.reach_goal_position(goal_dist, axis_dist, w_dist):
            self.is_acting[f'move_{object}'] = True
        pos_err = torch.where(goal_dist > self.goal_offset, pos - hand_pos, zero)
        pos_err = torch.where(xy_dist > xy_offset, torch.cat([pos_err[:, :2], torch.tensor([[0]], device=self.device)], -1), pos_err)
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(rot, hand_rot), zero)
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            if self.is_acting[f'move_{object}']:
                self.is_acting[f'move_{object}'] = False
                print(f"finish moving to {object}")
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
            hold_tight = 0.01 if "primary_tool" in self.env_cfg_dict.keys() else 0
            
            self.goal_pos_set = [
                torch.tensor([[tool_pos[:, 0], tool_pos[:, 1], tool_pos[:, 2] + 0.17]], device=self.device),
                torch.tensor([[tool_pos[:, 0], tool_pos[:, 1], tool_pos[:, 2] + 0.095 - hold_tight]], device=self.device),
                torch.tensor([[tool_pos[:, 0], tool_pos[:, 1], tool_pos[:, 2] + 0.13]], device=self.device),
                torch.tensor([[tool_pos[:, 0], tool_pos[:, 1] + 0.05, tool_pos[:, 2] + 0.13]], device=self.device),
                torch.tensor([[tool_pos[:, 0], tool_pos[:, 1] + 0.05, tool_pos[:, 2] + 0.2]], device=self.device)
            ]
            self.goal_rot_set = [torch.tensor([[rot.x, rot.y, rot.z, rot.w]], device=self.device)] * len(self.goal_pos_set)
            # if tool == 'knife':
            #     self.goal_rot_set[-1] = torch.tensor([[knife_rot.x, knife_rot.y, knife_rot.z, knife_rot.w]])
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
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        goal_pos = self.goal_pos_set[self.action_stage['take_tool']]
        goal_rot = self.goal_rot_set[self.action_stage['take_tool']]
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > self.goal_offset, goal_pos - hand_pos, torch.tensor([0, 0, 0], device=self.device))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.tensor([0, 0, 0], device=self.device))
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
                torch.tensor([[tool_pos.x, tool_pos.y + 0.05, tool_pos.z + 0.18]], device=self.device),
                torch.tensor([[tool_pos.x, tool_pos.y + 0.05, tool_pos.z + 0.15]], device=self.device),
                torch.tensor([[tool_pos.x, tool_pos.y - 0.0026, tool_pos.z + 0.13]], device=self.device),
                torch.tensor([[tool_pos.x, tool_pos.y - 0.0026, tool_pos.z + 0.085]], device=self.device),
                torch.tensor([[tool_pos.x, tool_pos.y - 0.0026, tool_pos.z + 0.2]], device=self.device)
            ]
            self.goal_rot_set = [torch.tensor([[rot.x, rot.y, rot.z, rot.w]], device=self.device)] * len(self.goal_pos_set)
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
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        goal_pos = self.goal_pos_set[self.action_stage['put_tool']]
        goal_rot = self.goal_rot_set[self.action_stage['put_tool']]
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > self.goal_offset, goal_pos - hand_pos, torch.tensor([0, 0, 0], device=self.device))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.tensor([0, 0, 0], device=self.device))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['put_tool'] += 1
            print(self.action_stage["put_tool"])
        if self.action_stage['put_tool'] > 3:
            self.pos_action[:, 7:9] = gripper_open
        return dpose * 0.8
        
    def move_around(self):
        
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        
        if self.action_stage['move_around'] == -1:
            # initialize
            print("move around start")
            self.delta['move_around'] = 1
            init_pos = hand_pos.clone()
            
            max_z = 0.81 if self.env_cfg_dict["primary_tool"] == "spoon" else 0.84
            min_z = 0.78 if self.env_cfg_dict["primary_tool"] == "spoon" else 0.81
            max_x = 0.62
            min_x = 0.32
            max_y = 0.38
            min_y = -0.23
            max_dist = 0.65
            
            # original pos: 0.3871, 0.0877, container pos: 0.43999999999999995, 0.07500000000000001
            self.goal_pos_set = []
            for i in np.arange(min_x, max_x, 0.05):
                for j in np.arange(min_y, max_y, 0.05):
                    if np.sqrt(i*i + j*j) < max_dist:
                        self.goal_pos_set.append(torch.tensor([[float(i), float(j), float(min_z + random.random() * (max_z - min_z))]], device=self.device))
            
                    
            self.goal_rot_set = [ torch.tensor([[ 0.9953,  0.0393, -0.0701,  0.0542]], device=self.device) for i in self.goal_pos_set]
            self.is_acting['move_around'] = True
            self.action_stage['move_around'] = 0
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        elif self.action_stage['move_around'] == len(self.goal_pos_set):
            # final stage
            if self.is_acting['move_around']:
                print("finish move_around")
                self.is_acting['move_around'] = False
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        goal_pos = self.goal_pos_set[self.action_stage['move_around']]
        goal_rot = self.goal_rot_set[self.action_stage['move_around']]
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > self.goal_offset, goal_pos - hand_pos, torch.tensor([0, 0, 0], device=self.device))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.tensor([0, 0, 0], device=self.device))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['move_around'] += 1
            print(self.action_stage['move_around'])
        dpose *= self.delta['move_around']
        return dpose
        
    def choose_action(self):
        # self.instruction += f" {len(self.action_sequence) + 1}. "
        rgb_path = os.path.join("observation", "rgb.png")
        depth_path = os.path.join("observation", "depth.png")
        rgb_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_COLOR).reshape(1080, 1920, 4)[:,:,:-1]
        depth_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_DEPTH)
        depth_image = np.clip(depth_image, -1.8, 0)
        depth_image = ((depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) * 255).astype(np.uint8)
        Image.fromarray(rgb_image).save(rgb_path)
        Image.fromarray(depth_image).save(depth_path)
        print(self.containers_list)
        combined_score = self.decision_pipeline.get_combine(self.instruction, rgb_path, depth_path, self.containers_list, self.tool_list, self.action_sequence, False)
        best_action = max(combined_score, key=combined_score.get)
        
        # self.instruction += best_action
        self.action_sequence.append(best_action)
        return best_action
    
    def find_nearest_container(self, init_pos):
        min_dist = float("inf")
        best_tensor = init_pos
        for indice in self.containers_indices.values():
            container_p = self.rb_state_tensor[indice, :3]
            container_p[..., -1] = 0
            print(f"container pos: {container_p}, init pos: {init_pos}")
            dist = torch.norm(container_p - init_pos, dim=1)
            if dist < min_dist:
                min_dist = dist
                best_tensor = container_p
        return best_tensor
    
    def scoop(self):
        
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        use_container_pos = True
        
        if self.action_stage['scoop'] == -1:
            # initialize
            
            print("scoop start")
            self.delta['scoop'] = [1.5, 1.5, 1.5, 1.5, 1.5, 3, 3, 3, 3, 3, 3, 2.5, 3, 3, 3, 3]
            # self.delta['scoop'] = [0.05, 0.05, 0.05, 0.5]
            self.goal_pos_set = [hand_pos + torch.tensor([[0., 0., -0.065]], device=self.device)]
            self.goal_rot_set = [torch.tensor([[1.0, 0.0, -0.05, 0.0]], device=self.device)]
            init_pos = hand_pos.clone()
            init_pos[:, 2] = 0
            
            # find the nearest container
            if use_container_pos:
                best_tensor = self.find_nearest_container(init_pos)
                init_pos = best_tensor - torch.tensor([[0.01, -0.02, 0.0]], device=self.device)
                init_pos[0][2] = 0.06
                init_pos[0][1] += 0.0127
                #init_pos[0][0] -= 0.02
            else:
                init_pos[0][2] -= 0.02
                
            
            # original pos: 0.3871, 0.0877, container pos: 0.43999999999999995, 0.07500000000000001
            self.goal_pos_set = [
                init_pos + torch.tensor([[0.0000, 0.0000, 0.7836]], device=self.device),
                init_pos + torch.tensor([[-0.0300,  0.0000,  0.7228]], device=self.device),
                init_pos + torch.tensor([[-0.0299,  0.0109,  0.6728]], device=self.device),
                init_pos + torch.tensor([[-0.0271,  0.0050,  0.6808]], device=self.device),
                init_pos + torch.tensor([[-0.0264,  0.0054,  0.6716]], device=self.device),
                init_pos + torch.tensor([[0.0169, 0.0043, 0.6700]], device=self.device),
                init_pos + torch.tensor([[-0.0288,  0.0043,  0.6638]], device=self.device),
                init_pos + torch.tensor([[-0.0295,  0.0043,  0.6638]], device=self.device),
                init_pos + torch.tensor([[-0.0300,  0.0038,  0.6650]], device=self.device),
                init_pos + torch.tensor([[-0.0352,  0.0035,  0.6668]], device=self.device),
                init_pos + torch.tensor([[-0.0505,  0.0031,  0.6698]], device=self.device),
                init_pos + torch.tensor([[-0.0807,  0.0022,  0.6728]], device=self.device),
                init_pos + torch.tensor([[-0.1024,  0.0019,  0.6758]], device=self.device),
                init_pos + torch.tensor([[-0.1016,  0.0021,  0.6708]], device=self.device),
                init_pos + torch.tensor([[-0.1370,  0.0019,  0.6808]], device=self.device),
                init_pos + torch.tensor([[-0.2130,  0.0017,  0.7000]], device=self.device)
            ]

            self.goal_rot_set = [
                torch.tensor([[ 0.9945,  0.0413, -0.0809,  0.0523]], device=self.device),
                torch.tensor([[ 0.9945,  0.0410, -0.0808,  0.0522]], device=self.device),
                torch.tensor([[ 0.9953,  0.0393, -0.0701,  0.0542]], device=self.device),
                torch.tensor([[ 0.9976,  0.0345, -0.0243,  0.0549]], device=self.device),
                torch.tensor([[ 0.9977,  0.0345, -0.0176,  0.0563]], device=self.device),
                torch.tensor([[0.9975, 0.0310, 0, 0.0575]], device=self.device),
                torch.tensor([[0.9975, 0.0310, 0.0288, 0.0575]], device=self.device),
                torch.tensor([[0.9965, 0.0388, 0.0500, 0.0580]], device=self.device),
                torch.tensor([[0.9952, 0.0278, 0.0736, 0.0586]], device=self.device),
                torch.tensor([[0.9925, 0.0257, 0.1034, 0.0594]], device=self.device),
                torch.tensor([[0.9869, 0.0225, 0.1478, 0.0604]], device=self.device),
                torch.tensor([[0.9621, 0.0144, 0.2648, 0.0626]], device=self.device),
                torch.tensor([[0.9340, 0.0089, 0.3514, 0.0637]], device=self.device),
                torch.tensor([[0.9337, 0.0093, 0.4071, 0.0639]], device=self.device),
                torch.tensor([[0.8848, 0.0023, 0.5116, 0.0641]], device=self.device),
                torch.tensor([[0.8844, 0.0025, 0.5116, 0.0640]], device=self.device)
            ]
            self.action_stage['scoop'] = 0
            self.is_acting['scoop'] = True
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        elif self.action_stage['scoop'] == len(self.goal_pos_set):
            # final stage
            if self.is_acting['scoop']:
                print("finish scoop")
                self.is_acting['scoop'] = False
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        goal_pos = self.goal_pos_set[self.action_stage['scoop']]
        goal_rot = self.goal_rot_set[self.action_stage['scoop']]
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > self.goal_offset, goal_pos - hand_pos, torch.tensor([0, 0, 0], device=self.device))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.tensor([0, 0, 0], device=self.device))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['scoop'] += 1
            print(self.action_stage['scoop'])
        dpose *= self.delta['scoop'][self.action_stage['scoop'] if self.action_stage['scoop'] < len(self.delta['scoop']) else -1]
        
        # print(dpose)
        return dpose
    
    def scoop_put(self):
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        use_container_pos = True
        
        if self.action_stage['scoop_put'] == -1:
            # initialize
            
            print("scoop_put start")
            self.delta['scoop_put'] = [1.8,1.8,2]
            # self.delta['scoop_put'] = [0.05, 0.05, 0.05, 0.5]
            self.goal_pos_set = [hand_pos + torch.tensor([[0., 0., -0.065]], device=self.device)]
            self.goal_rot_set = [torch.tensor([[1.0, 0.0, -0.05, 0.0]], device=self.device)]
            init_pos = hand_pos.clone()
            init_pos[:, 2] = 0
            
            # find the nearest container
            if use_container_pos:
                best_tensor = self.find_nearest_container(init_pos)
                init_pos = best_tensor - torch.tensor([[0.01, -0.02, 0.0]], device=self.device)
                init_pos[0][2] = 0.03
                init_pos[0][1] += 0.0127
                #init_pos[0][0] -= 0.02
            else:
                init_pos[0][2] -= 0.02
                
            
            # original pos: 0.3871, 0.0877, container pos: 0.43999999999999995, 0.07500000000000001
            x_shift = 0.03
            z_shift = 0.15
            self.goal_pos_set = [ 
                init_pos + torch.tensor([[-x_shift-0.1024,  0.0019,  0.6758+z_shift]], device=self.device), 
                init_pos + torch.tensor([[-x_shift-0.0807,  0.0022,  0.6728+z_shift]], device=self.device), 
                init_pos + torch.tensor([[-x_shift-0.0505,  0.0031,  0.6698+z_shift]], device=self.device), 
                init_pos + torch.tensor([[-x_shift-0.0352,  0.0035,  0.6668+z_shift]], device=self.device), 
                init_pos + torch.tensor([[-x_shift-0.0300,  0.0038,  0.6650+z_shift]], device=self.device), 
                init_pos + torch.tensor([[-x_shift-0.0295,  0.0043,  0.6638+z_shift]], device=self.device), 
                init_pos + torch.tensor([[-x_shift-0.0288,  0.0043,  0.6638+z_shift]], device=self.device), 
                init_pos + torch.tensor([[-x_shift+0.0169, 0.0043, 0.6700+z_shift]], device=self.device), 
                init_pos + torch.tensor([[-x_shift+0.0000, 0.0000, 0.7836+z_shift]], device=self.device)
            ]
            self.goal_rot_set = [ 
                torch.tensor([[0.9340, 0.0089, 0.3514, 0.0637]], device=self.device), 
                torch.tensor([[0.9621, 0.0144, 0.2648, 0.0626]], device=self.device), 
                torch.tensor([[0.9869, 0.0225, 0.1478, 0.0604]], device=self.device), 
                torch.tensor([[0.9925, 0.0257, 0.1034, 0.0594]], device=self.device), 
                torch.tensor([[0.9952, 0.0278, 0.0736, 0.0586]], device=self.device), 
                torch.tensor([[0.9965, 0.0388, 0.0500, 0.0580]], device=self.device), 
                torch.tensor([[0.9975, 0.0310, 0.0288, 0.0575]], device=self.device), 
                torch.tensor([[0.9975, 0.0310, 0.0000, 0.0575]], device=self.device),
                torch.tensor([[ 0.9945,  0.0413, -0.0809,  0.0523]], device=self.device)
            ]
            self.action_stage['scoop_put'] = 0
            self.is_acting['scoop_put'] = True
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        elif self.action_stage['scoop_put'] == len(self.goal_pos_set):
            # final stage
            if self.is_acting['scoop_put']:
                print("finish scoop_put")
                self.is_acting['scoop_put'] = False
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        goal_pos = self.goal_pos_set[self.action_stage['scoop_put']]
        goal_rot = self.goal_rot_set[self.action_stage['scoop_put']]
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > self.goal_offset, goal_pos - hand_pos, torch.tensor([0, 0, 0], device=self.device))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.tensor([0, 0, 0], device=self.device))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['scoop_put'] += 1
            print(self.action_stage['scoop_put'])
        dpose *= self.delta['scoop_put'][self.action_stage['scoop_put'] if self.action_stage['scoop_put'] < len(self.delta['scoop_put']) else -1]
        
        return dpose
    
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        delta = 0.5
        
        if self.action_stage['scoop_put'] == -1:
            goal_rot = torch.tensor([[1, 0, -0.1, 0]])
            stage_num = 2
            use_container_pos = True
            # find the nearest container
            if use_container_pos:
                min_dist = float('inf')
                for i, container_pose in enumerate(self.containers_pose):
                    container_p = torch.tensor([[container_pose.p.x, container_pose.p.y, 0.0]])
                    print(f"container pos: {container_p}, init pos: {init_pos}")
                    dist = torch.norm(container_p - init_pos, dim=1)
                    if dist < min_dist:
                        min_dist = dist
                        best_tensor = container_p
                init_pos = best_tensor - torch.tensor([[0.01, -0.02, 0.0]])
                init_pos[0][2] = 0.03
                init_pos[0][1] += 0.0127
                #init_pos[0][0] -= 0.02
            else:
                init_pos[0][2] -= 0.02
            # if use_container_pos:
            #     init_pos = hand_pos.clone()
            #     init_pos[:, 2] = 0
            #     min_dist = float('inf')
            #     for i, container_pose in enumerate(self.containers_pose):
            #         container_p = torch.tensor([[container_pose.p.x, container_pose.p.y, 0.0]])
            #         print(f"container pos: {container_p}, init pos: {init_pos}")
            #         dist = torch.norm(container_p - init_pos, dim=1)
            #         if dist < min_dist:
            #             min_dist = dist
            #             best_tensor = container_p
            #     init_pos = best_tensor - torch.tensor([[0.01, -0.02, 0.0]])
            #     init_pos[0][2] = 0.03
            #     init_pos[0][1] += 0.0127
            
            self.goal_pos_set = [
                init_pos + torch.tensor([[-0.2130,  0.0017,  0.7000]]), 
                init_pos + torch.tensor([[-0.1370,  0.0019,  0.6808]]), 
                init_pos + torch.tensor([[-0.1016,  0.0021,  0.6708]]), 
                init_pos + torch.tensor([[-0.1024,  0.0019,  0.6758]]), 
                init_pos + torch.tensor([[-0.0807,  0.0022,  0.6728]]), 
                init_pos + torch.tensor([[-0.0505,  0.0031,  0.6698]]), 
                init_pos + torch.tensor([[-0.0352,  0.0035,  0.6668]]), 
                init_pos + torch.tensor([[-0.0300,  0.0038,  0.6650]]), 
                init_pos + torch.tensor([[-0.0295,  0.0043,  0.6638]]), 
                init_pos + torch.tensor([[-0.0288,  0.0043,  0.6638]]), 
                init_pos + torch.tensor([[0.0169, 0.0043, 0.6700]]), 
                init_pos + torch.tensor([[-0.0264,  0.0054,  0.6716]]), 
                init_pos + torch.tensor([[-0.0271,  0.0050,  0.6808]]), 
                init_pos + torch.tensor([[-0.0299,  0.0109,  0.6728]]), 
                init_pos + torch.tensor([[-0.0300,  0.0000,  0.7228]]), 
                init_pos + torch.tensor([[0.0000, 0.0000, 0.7836]])
            ]
            self.goal_rot_set = [
                torch.tensor([[0.8844, 0.0025, 0.5116, 0.0640]]), 
                torch.tensor([[0.8848, 0.0023, 0.5116, 0.0641]]), 
                torch.tensor([[0.9337, 0.0093, 0.4071, 0.0639]]), 
                torch.tensor([[0.9340, 0.0089, 0.3514, 0.0637]]), 
                torch.tensor([[0.9621, 0.0144, 0.2648, 0.0626]]), 
                torch.tensor([[0.9869, 0.0225, 0.1478, 0.0604]]), 
                torch.tensor([[0.9925, 0.0257, 0.1034, 0.0594]]), 
                torch.tensor([[0.9952, 0.0278, 0.0736, 0.0586]]), 
                torch.tensor([[0.9965, 0.0388, 0.0500, 0.0580]]), 
                torch.tensor([[0.9975, 0.0310, 0.0288, 0.0575]]), 
                torch.tensor([[0.9975, 0.0310, 0.0000, 0.0575]]), 
                torch.tensor([[ 0.9977,  0.0345, -0.0176,  0.0563]]), 
                torch.tensor([[ 0.9976,  0.0345, -0.0243,  0.0549]]), 
                torch.tensor([[ 0.9953,  0.0393, -0.0701,  0.0542]]), 
                torch.tensor([[ 0.9945,  0.0410, -0.0808,  0.0522]]), 
                torch.tensor([[ 0.9945,  0.0413, -0.0809,  0.0523]])
            ]
            # self.goal_rot_set.extend([
            #     hand_rot + goal_rot * (i + 1) for i in range(stage_num)
            # ])
            self.is_acting['scoop_put'] = True
            self.action_stage['scoop_put'] = 0
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        
        elif self.action_stage['scoop_put'] == len(self.goal_rot_set):
            # final stage
            print("finish putting")
            self.is_acting['scoop_put'] = False
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        
        # if self.action_stage['scoop_put'] == 0:
            
        to_goal = self.goal_pos_set[0] - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        pos_err = torch.where(goal_dist > self.goal_offset, to_goal, torch.tensor([0, 0, 0]))
        if goal_dist < self.goal_offset:
            self.action_stage['scoop_put'] += 1
        # return torch.cat([pos_err, torch.tensor([[0, 0, 0]])], -1).unsqueeze(-1) * 0.3
        
        goal_rot = self.goal_rot_set[self.action_stage['scoop_put']]
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.tensor([[0, 0, 0]])
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.tensor([0, 0, 0]))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        if axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['scoop_put'] += 1
            print(self.action_stage['scoop_put'])
        dpose *= delta
        return dpose
        
    def stir(self):
        
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        use_container_pos = True
        
        
        if self.action_stage['stir'] == -1:
            if use_container_pos:
                init_pos = hand_pos.clone()
                best_tensor = self.find_nearest_container(init_pos)
                init_pos = best_tensor + torch.tensor([[-0.05, 0.0, 0.4]], device=self.device)
            stage_num = 10
            self.stir_center = init_pos + torch.tensor([[0.04, 0, 0]], device=self.device)
            radius = 0.04
            x = -0.04
            middle = False
            step = radius * 4 / stage_num
            self.goal_pos_set = [init_pos, init_pos + torch.tensor([0, 0, -0.17], device=self.device)]
            print(self.goal_pos_set)
            for _ in range(stage_num):
                x += step if not middle else -step
                y = math.sqrt(radius ** 2 - x ** 2)
                y *= 1 if not middle else -1
                if x >= radius and not middle:
                    middle = True
                self.goal_pos_set.append(self.stir_center + torch.tensor([x, y, 0], device=self.device))
            self.goal_pos_set.append(self.goal_pos_set[-1] + torch.tensor([[0., 0., 0.1]], device=self.device))
                
            self.action_stage['stir'] = 0
            self.is_acting['stir'] = True
            self.delta['stir'] = 0.05
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
            
        elif self.action_stage['stir'] == len(self.goal_pos_set):
            if self.is_acting['stir']:
                print("finish stirring")
                self.is_acting['stir'] = False
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        if self.action_stage['stir'] <= 1 or self.action_stage['stir'] == len(self.goal_pos_set) - 1:
            to_goal = self.goal_pos_set[self.action_stage['stir']] - hand_pos
            goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
            pos_err = torch.where(goal_dist > self.goal_offset, to_goal, torch.tensor([0, 0, 0], device=self.device))
            if goal_dist < self.goal_offset:
                print(self.action_stage['stir'])
                self.action_stage['stir'] += 1
            return torch.cat([pos_err, torch.tensor([[0, 0, 0]], device=self.device)], -1).unsqueeze(-1) * 0.5
        
        stage_xy = self.goal_pos_set[self.action_stage['stir']]
        stage_dir = torch.sub(stage_xy, hand_pos)
        stage_dist = torch.norm(stage_dir[:, :1], dim=1).unsqueeze(-1) 
        if stage_dist < self.goal_offset:
            self.action_stage['stir'] += 1
        
        container_dir = self.stir_center - hand_pos
        container_dist = torch.norm(container_dir[:, :1], dim=1).unsqueeze(-1)
        stir_radius = 1
        if container_dist > stir_radius:
            print("reset", container_dir, container_dist)
            self.is_acting['stir'] = False
            return torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]], device=self.device)
        goal_xy = container_dir[:, [1, 0, 2]] * torch.tensor([[-1, 1, 0]], device=self.device)
        goal_norm = torch.norm(goal_xy, dim=1).unsqueeze(-1)
        goal_xy /= goal_norm
        pos_err = goal_xy
        orn_err = torch.tensor([[0, 0, 0]], device=self.device)
        
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1) * self.delta["stir"]    
        return dpose
        
    def fork(self):
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        self.delta['fork'] = 0.3
        if self.action_stage['fork'] == -1:
            food_pos = self.rb_state_tensor[self.forked_food_indices, :3]
            food_rot = self.rb_state_tensor[self.forked_food_indices, 3:7]
            min_dist = float('inf')
            init_pos = hand_pos
            for indice in self.forked_food_indices:
                food_pos = self.rb_state_tensor[indice, :3]
                food_rot = self.rb_state_tensor[indice, 3:7]
                print(f"food pos: {food_pos}")
                dist = torch.norm(food_pos - init_pos, dim=1)
                if dist < min_dist:
                    min_dist = dist
                    best_tensor = [food_pos, food_rot]
            init_pos = best_tensor[0] + torch.tensor([[0.01, -0.0145, 0.4]], device=self.device) # torch.tensor([[0.003, 0.003, 0.4]])
            
            init_rot = best_tensor[1]
            roll, pitch, yaw = quaternion_to_euler(gymapi.Quat(init_rot[0], init_rot[1], init_rot[2], init_rot[3]))
            roll += 1.57
            yaw = math.pi / 2
            init_rot = euler_to_quaternion(roll, pitch, yaw, True)
            self.goal_pos_set = [
                init_pos,
                init_pos + torch.tensor([[0, 0, -0.25]], device=self.device)
            ]
            self.goal_rot_set = [init_rot] * len(self.goal_pos_set)
            print(f'init pose: {init_pos} {init_rot}')
            self.is_acting['fork'] = True
            self.action_stage['fork'] = 0
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        elif self.action_stage['fork'] == len(self.goal_pos_set):
            if self.is_acting['fork']:
                print("finish forking")
                self.is_acting['fork'] = False
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        fork_offset = 3e-3
        goal_pos = self.goal_pos_set[self.action_stage['fork']]
        goal_rot = self.goal_rot_set[self.action_stage['fork']]
        
        to_goal = goal_pos - hand_pos
        if self.action_stage['fork'] == 1:
            to_goal = to_goal * torch.tensor([[0, 0, 0.5]], device=self.device)
        goal_dist = torch.norm(to_goal[..., :2], dim=1).unsqueeze(-1) if self.action_stage['fork'] == 0 else abs(to_goal[..., 2])
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > fork_offset, to_goal, torch.tensor([0, 0, 0], device=self.device))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.tensor([0, 0, 0]))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        if goal_dist <= fork_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['fork'] += 1
        dpose *= self.delta['fork']
        return dpose
    
    def get_action_initial_state(self, action):
        """Initailize trajectory and some state of action
            Including self.goal_pos_set, self.goal_rot_set, is_acting flag
        Args:
            action (str): Name of action
        """
        assert action in self.action_list, f"Unsupported Action {action}"
        
        hand_pos:torch.tensor = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot:torch.tensor = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        
        if self.use_container_pos:
            min_dist = float('inf')
            for i, container_pose in enumerate(self.containers_pose):
                container_p = torch.tensor([[container_pose.p.x, container_pose.p.y, container_pose.p.z]], device=self.device)
                print(f"container pos: {container_p}, init pos: {init_pos}")
                dist = torch.norm(container_p - init_pos, dim=1)
                if dist < min_dist:
                    min_dist = dist
                    best_tensor = container_p
            init_pos = best_tensor
            init_pos -= torch.tensor([[0., 0., 0.]], device=self.device)
        else:
            init_pos = hand_pos.clone()
            init_pos -= torch.tensor([[0., 0., 0.]], device=self.device)
                
        self.goal_pos_set = []
        self.goal_rot_set = []
        self.is_acting[action] = True
        
        
        pass
    
    def do_action(self, action) -> torch.tensor:
        """
        TODO
        Merge all action in one function

        Args:
            action (str): Name of action

        Returns:
            torch.tensor: dpose
        """
        assert action in self.action_list, f"Unsupported Action {action}"
        
        hand_pos:torch.tensor = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot:torch.tensor = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        use_container_pos = True
        no_move = torch.zeros((6, 1), dtype=torch.float32)
        
        if self.action_stage[action] == -1:
            if self.use_container_pos:
                min_dist = float('inf')
                for i, container_pose in enumerate(self.containers_pose):
                    container_p = torch.tensor([[container_pose.p.x, container_pose.p.y, container_pose.p.z]], device=self.device)
                    print(f"container pos: {container_p}, init pos: {init_pos}")
                    dist = torch.norm(container_p - init_pos, dim=1)
                    if dist < min_dist:
                        min_dist = dist
                        best_tensor = container_p
                init_pos = best_tensor
                init_pos -= torch.tensor([[0., 0., 0.]], device=self.device)
            else:
                init_pos = hand_pos.clone()
                init_pos -= torch.tensor([[0., 0., 0.]], device=self.device)
            
            self.action_stage[action] = 0
            self.is_acting[action] = True
            get_action_state = getattr(action_state, action)
            self.goal_pos_set, self.goal_rot_set = get_action_state(init_pos)
            return no_move
        
        elif self.action_stage[action] == len(self.goal_pos_set):
            if self.is_acting[action]:
                print(f'Finish {action}')
                self.is_acting[action] = False
            return no_move
        
        current_action_stage = self.action_stage[action]
        goal_pos = self.goal_pos_set[current_action_stage]
        goal_rot = self.goal_rot_set[current_action_stage]
        
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > self.goal_offset, goal_pos - hand_pos, torch.tensor([0, 0, 0], device=self.device))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.tensor([0, 0, 0], device=self.device))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage[action] += 1
            print(self.action_stage[action])
        dpose *= self.delta[action][current_action_stage if current_action_stage < len(self.delta[action]) else -1]
        
        return dpose

        
        if self.action_stage['scoop'] == -1:
            # initialize
            
            print("scoop start")
            self.delta['scoop'] = [0.5, 0.5, 0.5, 0.5]
            # self.delta['scoop'] = [0.05, 0.05, 0.05, 0.5]
            self.goal_pos_set = [hand_pos + torch.tensor([[0., 0., -0.065]])]
            self.goal_rot_set = [torch.tensor([[1.0, 0.0, -0.05, 0.0]])]
            init_pos = hand_pos.clone()
            init_pos[:, 2] = 0
            
            # find the nearest container
            if use_container_pos:
                min_dist = float('inf')
                for i, container_pose in enumerate(self.containers_pose):
                    container_p = torch.tensor([[container_pose.p.x, container_pose.p.y, 0.0]])
                    print(f"container pos: {container_p}, init pos: {init_pos}")
                    dist = torch.norm(container_p - init_pos, dim=1)
                    if dist < min_dist:
                        min_dist = dist
                        best_tensor = container_p
                init_pos = best_tensor - torch.tensor([[0.01, -0.02, 0.0]])
                init_pos[0][2] = 0.03
                init_pos[0][1] += 0.0127
                #init_pos[0][0] -= 0.02
            else:
                init_pos[0][2] -= 0.02
                
            
            # original pos: 0.3871, 0.0877, container pos: 0.43999999999999995, 0.07500000000000001
            self.goal_pos_set = [
                init_pos + torch.tensor([[0.0000, 0.0000, 0.7836]]),
                init_pos + torch.tensor([[-0.0300,  0.0000,  0.7228]]),
                init_pos + torch.tensor([[-0.0299,  0.0109,  0.6728]]),
                init_pos + torch.tensor([[-0.0271,  0.0050,  0.6808]]),
                init_pos + torch.tensor([[-0.0264,  0.0054,  0.6716]]),
                init_pos + torch.tensor([[0.0169, 0.0043, 0.6700]]),
                init_pos + torch.tensor([[-0.0288,  0.0043,  0.6638]]),
                init_pos + torch.tensor([[-0.0295,  0.0043,  0.6638]]),
                init_pos + torch.tensor([[-0.0300,  0.0038,  0.6650]]),
                init_pos + torch.tensor([[-0.0352,  0.0035,  0.6668]]),
                init_pos + torch.tensor([[-0.0505,  0.0031,  0.6698]]),
                init_pos + torch.tensor([[-0.0807,  0.0022,  0.6728]]),
                init_pos + torch.tensor([[-0.1024,  0.0019,  0.6758]]),
                init_pos + torch.tensor([[-0.1016,  0.0021,  0.6708]]),
                init_pos + torch.tensor([[-0.1370,  0.0019,  0.6808]]),
                init_pos + torch.tensor([[-0.2130,  0.0017,  0.7000]])
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
                torch.tensor([[0.9952, 0.0278, 0.0736, 0.0586]]),
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
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        
        elif self.action_stage['scoop'] == len(self.goal_pos_set):
            # final stage
            if self.is_acting['scoop']:
                print("finish scoop")
                self.is_acting['scoop'] = False
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]])
        
        goal_pos = self.goal_pos_set[self.action_stage['scoop']]
        goal_rot = self.goal_rot_set[self.action_stage['scoop']]
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > self.goal_offset, goal_pos - hand_pos, torch.tensor([0, 0, 0]))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.tensor([0, 0, 0]))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['scoop'] += 1
            print(self.action_stage['scoop'])
        dpose *= self.delta['scoop'][self.action_stage['scoop'] if self.action_stage['scoop'] < len(self.delta['scoop']) else -1]
        
        return dpose
        
    def cut(self):
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        
        
        if self.action_stage['cut'] == -1:
            self.delta["cut"] = [0.5]
            self.action_stage['cut'] = 0
            self.is_acting['cut'] = True
            food_pos = torch.mean(self.rb_state_tensor[self.butter_indices, :3], dim=0)
            food_rot = self.rb_state_tensor[self.butter_indices, 3:7][0]
            print(food_rot)
            roll, pitch, yaw = quaternion_to_euler(gymapi.Quat(food_rot[..., 0], food_rot[..., 1], food_rot[..., 2], food_rot[..., 3]))
            roll_init = roll + 1.57
            pitch_init = pitch - math.pi / 3
            init_rot = euler_to_quaternion(roll_init, pitch_init, yaw)
            final_rot = euler_to_quaternion(3.14, 0, 0)
            print(food_pos)
            self.goal_pos_set = [
                food_pos + torch.tensor([[-0.18, -0.0189, 0.3]], device=self.device),
                food_pos + torch.tensor([[-0.18, -0.0189, 0.1]], device=self.device),
                food_pos + torch.tensor([[-0.18, -0.0189, 0.3]], device=self.device),
            ]
            self.goal_rot_set = [
                torch.tensor([[init_rot.x, init_rot.y, init_rot.z, init_rot.w]], device=self.device),
                torch.tensor([[init_rot.x, init_rot.y, init_rot.z, init_rot.w]], device=self.device),
                torch.tensor([[final_rot.x, final_rot.y, final_rot.z, final_rot.w]], device=self.device)
            ]
            return torch.tensor([[[0.], [0.], [0.], [0.], [0.], [0.]]], device=self.device)
        elif self.action_stage["cut"] == len(self.goal_pos_set):
            if self.is_acting["cut"]:
                print("finish cut")
                self.action_stage['cut'] = -1
                self.is_acting['cut'] = False
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        goal_pos = self.goal_pos_set[self.action_stage['cut']]
        goal_rot = self.goal_rot_set[self.action_stage['cut']]
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > self.goal_offset, goal_pos - hand_pos, torch.tensor([0, 0, 0], device=self.device))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.tensor([0, 0, 0], device=self.device))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['cut'] += 1
            print(self.action_stage['cut'])
        dpose *= self.delta['cut'][self.action_stage['cut'] if self.action_stage['cut'] < len(self.delta['cut']) else -1]
        
        return dpose
    
    def pull_bowl_closer(self):
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        use_container_pos = True
        gripper_open = self.franka_dof_upper_limits[7:]
        gripper_close = self.franka_dof_lower_limits[7:]
        if self.action_stage['pull_bowl_closer'] == -1:
            # initialize
            
            print("pull bowl closer start")
            self.delta['pull_bowl_closer'] = [0.5, 0.5, 0.5, 0.4, 0.5, 0.5]
            self.goal_pos_set = [hand_pos + torch.tensor([[0., 0., -0.065]], device=self.device)]
            self.goal_rot_set = [torch.tensor([[1.0, 0.0, -0.05, 0.0]], device=self.device)]
            init_pos = hand_pos.clone()
            init_pos[:, 2] = 0
            
            # find the nearest container
            if use_container_pos:
                best_tensor = self.find_nearest_container(init_pos)
                init_pos = best_tensor - torch.tensor([[0.01, -0.02, 0.0]], device=self.device)
                init_pos[0][2] = 0.03
                init_pos[0][1] += 0.0127
            else:
                init_pos[0][2] -= 0.02
                
            self.goal_pos_set = [
                init_pos + torch.tensor([[-0.07, -0.07, 0.7836]], device=self.device),
                init_pos + torch.tensor([[-0.07, -0.07,  0.57]], device=self.device),
                init_pos + torch.tensor([[-0.07, -0.07,  0.57]], device=self.device),
                # init_pos + torch.tensor([[-0.2, -0.3,  0.57]], device=self.device)
                torch.tensor([[0.3,-0.1,0.63]], device=self.device),
                torch.tensor([[0.3,-0.1,0.63]], device=self.device),
                torch.tensor([[0.3,-0.1,0.8]], device=self.device),
            ]

            self.goal_rot_set = [
                torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
                torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
                torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
                torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
                torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
                torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device)
            ]
            self.action_stage['pull_bowl_closer'] = 0
            self.is_acting['pull_bowl_closer'] = True
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        elif self.action_stage['pull_bowl_closer'] == len(self.goal_pos_set):
            # final stage
            if self.is_acting['pull_bowl_closer']:
                print("finish pull_bowl_closer")
                self.is_acting['pull_bowl_closer'] = False
            return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
        
        if self.action_stage['pull_bowl_closer'] > 1 and self.action_stage['pull_bowl_closer'] < 5:
            self.pos_action[:, 7:9] = gripper_close
        else:
            self.pos_action[:, 7:9] = gripper_open
        goal_pos = self.goal_pos_set[self.action_stage['pull_bowl_closer']]
        goal_rot = self.goal_rot_set[self.action_stage['pull_bowl_closer']]
        to_goal = goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > self.goal_offset, goal_pos - hand_pos, torch.tensor([0, 0, 0], device=self.device))
        orn_err = torch.where(axis_dist > self.axis_offset or w_dist > self.w_offset, self.orientation_error(goal_rot, hand_rot), torch.tensor([0, 0, 0], device=self.device))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        if goal_dist <= self.goal_offset and axis_dist <= self.axis_offset and w_dist <= self.w_offset:
            self.action_stage['pull_bowl_closer'] += 1
            print(self.action_stage['pull_bowl_closer'])
        dpose *= self.delta['pull_bowl_closer'][self.action_stage['pull_bowl_closer'] if self.action_stage['pull_bowl_closer'] < len(self.delta['pull_bowl_closer']) else -1]
        
        return dpose
    
    def test_llm(self):
        self.reset()
        test_time_limit = 10 # sec
        start_wait = 2
        start = time()
        self.action_start = time()
        self.executing = False
        self.best_action = None
        action_seq = []
        
        while not self.gym.query_viewer_has_closed(self.viewer):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.render_all_camera_sensors(self.sim)

            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            
            
            if time() - start > start_wait:
                self.best_action = self.choose_action()
                print(self.best_action)
                action_seq.append(self.best_action)
            if self.best_action == "DONE":
                break
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            self.frame += 1

        print(self.instruction)
        print('\n'.join([f'{i+1}. {action}' for i, action in enumerate(action_seq)]))
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
    
    def test_pipeline(self):
        self.reset()
        execute_time_limit = 200 # sec
        start_wait = 2
        start = time()
        self.action_start = time()
        self.executing = False
        self.best_action = None
        
        while not self.gym.query_viewer_has_closed(self.viewer):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.render_all_camera_sensors(self.sim)

            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            
            
            if not self.executing and time() - start > start_wait:
                sleep(20)
                self.best_action = self.choose_action()
                print(self.best_action)
                self.executing = True

            dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]], device=self.device)

            if self.best_action is None:
                pass
            elif self.best_action == "scoop":
                dpose = self.scoop()
            elif self.best_action == "stir":
                dpose = self.stir()
            elif self.best_action == "fork":
                dpose = self.fork()
            elif self.best_action == "cut":
                dpose = self.cut()
            elif self.best_action == "put_food":
                dpose = self.scoop_put()
            elif self.best_action == "DONE":
                break 
            elif "take_tool" in self.best_action:
                for tool in self.tool_list:
                    if tool in self.best_action:
                        dpose = self.take_tool(tool)
                        break
            elif "put_tool" in self.best_action:
                for tool in self.tool_list:
                    if tool in self.best_action:
                        dpose = self.put_tool(tool)
                        break
            elif "move" in self.best_action:
                for object in self.containers_list:
                    if object.split()[0] in self.best_action:
                        dpose = self.move(object, slow=True)
                        break
            if self.best_action and time() - self.action_start > execute_time_limit or not True in self.is_acting.values():
                # print(f"{self.best_action} done")s
                self.executing = False
                self.action_state_reset()
                self.action_start = time()
            
            
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
            
    def data_collection(self):
        self.reset()
        self.last_action = ""

        img_num = 0
        start = time()
        image_freq = 1.5 # sec.

        action = ""
        current_tool = "spoon"
        
        while not self.gym.query_viewer_has_closed(self.viewer):
            if self.container_num > 0:
                container_pos = [self.rb_state_tensor[indice, :3] for indice in self.containers_indices.values()]
                scoop_idx = 0
                put_idx = -1
                scoop_pos = container_pos[scoop_idx] + torch.tensor([[-0.1, 0., 0.3]], device=self.device)
                scoop_rot = torch.tensor([[1., 0., 0., 0.]], device=self.device)
                scoop_put_pos = container_pos[put_idx] + torch.tensor([[-0.15, 0., 0.3]], device=self.device)
                stir_pos = container_pos[scoop_idx] + torch.tensor([[-0.06, 0., 0.2]], device=self.device)
                stir_rot = torch.tensor([[1., 0., 0., 0.]], device=self.device)
            # scoop_rot = quat_from_angle_axis(torch.tensor([0.3]), to_container)
            # scoop_rot = torch.cat([to_container.view(3, ), torch.tensor([0.8])], -1).unsqueeze(-1).view(1, 4)
            
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

            for evt in self.gym.query_viewer_action_events(self.viewer):
                action = evt.action if (evt.value) > 0 else ""
            action != "" and self.last_action != action and print(action)
            
            if action == "action_reset":
                self.action_state_reset()
            elif action == "quit":
                break
            
            for a in self.action_list:
                if self.is_acting[a]:
                    if action in [a, ""]:
                        action = a
                    elif action != "save":
                        self.action_state_reset()
                        print(f"{a} terminate")
                    break
                
            if time() - start > image_freq:
                for i in range(self.num_envs):
                    file_name = f'./observation/{self.output_folder}/image{img_num}.png'
                    img_num += 1
                    self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR, file_name)
                    start = time()
            if "move_" in action:
                destination = action.replace("move_", "")
                if destination == "around":
                    self.move_around()
                else:
                    dpose = self.move(destination)
            elif action == "up":
                dpose = torch.tensor([[[0.],[0.],[1.],[0.],[0.],[0.]]]) * delta
            elif action == "down":
                dpose = torch.tensor([[[0.],[0.],[-1.],[0.],[0.],[0.]]]) * delta
            elif action == "left":
                dpose = torch.tensor([[[0.],[-1.],[0.],[0.],[0.],[0.]]]) * delta
            elif action == "right":
                dpose = torch.tensor([[[0.],[1.],[0.],[0.],[0.],[0.]]]) * delta
            elif action == "backward":
                dpose = torch.tensor([[[-1.],[0.],[0.],[0.],[0.],[0.]]]) * delta
            elif action == "forward":
                dpose = torch.tensor([[[1.],[0.],[0.],[0.],[0.],[0.]]]) * delta
            elif action == "turn_left":
                dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[-10.]]]) * delta
            elif action == "turn_right":
                dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[10.]]]) * delta
            elif action == "turn_up":
                dpose = torch.tensor([[[0.],[0.],[0.],[0.],[10.],[0.]]]) * delta
            elif action == "turn_down":
                dpose = torch.tensor([[[0.],[0.],[0.],[0.],[-10.],[0.]]]) * delta
            elif action == "test_pos":
                dpose = torch.tensor([[[0.],[0.],[0.],[1.],[0.],[0.]]]) * delta
            elif action == "test_neg":
                dpose = torch.tensor([[[0.],[0.],[0.],[-1.],[0.],[0.]]]) * delta
            elif action == "scoop":
                dpose = self.scoop()
            elif action == "stir":
                dpose = self.stir()
            elif action == "scoop_put":
                dpose = self.scoop_put()
            elif action == "fork":
                dpose = self.fork()
            elif action == "cut":
                dpose = self.cut()
            elif action == "take_tool":
                dpose = self.take_tool(current_tool)
            elif action == "put_tool":
                dpose = self.put_tool(current_tool)
            elif action == "move_around":
                dpose = self.move_around()
            elif action == "pull_bowl_closer":
                dpose = self.pull_bowl_closer()
            elif action == "gripper_close":
                dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
                if torch.all(self.pos_action[:, 7:9] == gripper_close):
                    self.pos_action[:, 7:9] = gripper_open
                elif torch.all(self.pos_action[:, 7:9] == gripper_open):
                    self.pos_action[:, 7:9] = gripper_close
            elif action == "set_tool":
                for i, tool in enumerate(self.tool_list):
                    if tool == current_tool:
                        current_tool = self.tool_list[(i + 1) % len(self.tool_list)]
                        print(current_tool)
                        break
                dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
            elif action == "choose action":
                self.choose_action()
                dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
                
            elif action == "save":
                hand_pos = self.rb_state_tensor[self.franka_hand_indices, 0:3]
                hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
                dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
                print(hand_pos)
                print(hand_rot)
                print(self.rb_state_tensor[self.franka_hand_indices, 7:])
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
            else:
                dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
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
            self.last_action = action

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def simulate(self):
        self.reset()
        grasp_pos = torch.tensor([[ 0, 0,  0]], dtype=torch.float32, device=self.device) # [ 0.5064, -0.1349,  0.4970]
        grasp_rot = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32, device=self.device)

        stage = 0

        while not self.gym.query_viewer_has_closed(self.viewer):
            print(self.containers_list)
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
        
    def executable(self):
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_pos = hand_pos[0]
        if self.env_cfg_dict['primary_tool'] == 'spoon':
            for i in range(len(self.containers_pose)):
                if self.env_cfg_dict['containers'][i]['food'] == 'ball':
                    distance = np.sqrt((hand_pos[0] - torch.tensor(self.containers_pose[i].p.x))*(hand_pos[0] - torch.tensor(self.containers_pose[i].p.x)) + (hand_pos[1] - torch.tensor(self.containers_pose[i].p.y))*(hand_pos[1] - torch.tensor(self.containers_pose[i].p.y)))
                    if distance < 0.1:
                        return True
        else:
            for i in range(len(self.containers_pose)):
                if self.env_cfg_dict['containers'][i]['food'] == 'cuttable_food' or self.env_cfg_dict['containers'][i]['food'] == "forked_food":
                    
                    distance = np.sqrt((hand_pos[0] - torch.tensor(self.containers_pose[i].p.x))*(hand_pos[0] - torch.tensor(self.containers_pose[i].p.x)) + (hand_pos[1] - torch.tensor(self.containers_pose[i].p.y))*(hand_pos[1] - torch.tensor(self.containers_pose[i].p.y)))
                    if distance < 0.09:
                        return True
        
        return False
    
    def new_config(self, config):
        self.__init__(config)
    
    def take_tool_move_around(self):
        self.reset()
        start = time()
        image_freq = 1.5 # sec.
                
        dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
        action_sequence = ['action_reset', 'take_tool', 'move_around']
        action_idx = 0

        
        rgb_images_0 = []
        depth_images_0 = []
        rgb_images_1 = []
        depth_images_1 = []
        while 1:

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)

            action = action_sequence[action_idx]

            if action == "action_reset":
                self.action_state_reset()
                action_idx += 1
            
            if time() - start > image_freq:
                rgb_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_COLOR).reshape(1080, 1920, 4)[:,:,:-1]
                depth_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_DEPTH)
                depth_image = np.clip(depth_image, -1.8, 0)
                depth_image = ((depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) * 255).astype(np.uint8)
                
                if not self.executable():
                    rgb_images_0.append(rgb_image)
                    depth_images_0.append(depth_image)
                else:
                    rgb_images_1.append(rgb_image)
                    depth_images_1.append(depth_image)
                
                
                start = time()
                
            
            if action == "take_tool":
                dpose = self.take_tool(self.env_cfg_dict["primary_tool"])
                if self.is_acting['take_tool'] == False:
                    action_idx += 1
            
            elif action == "move_around":
                dpose = self.move_around()
                if self.is_acting['move_around'] == False:
                    return rgb_images_0, depth_images_0, rgb_images_1, depth_images_1
                        
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

    def reinit(self, env_cfg_dict):
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

def read_yaml(file_path, env_type='medium_env', env_num=1):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    config = config[env_type][env_num]
    return config



if __name__ == "__main__":
    config = read_yaml("config.yaml", env_type='simple', env_num=6)
    issac = IsaacSim(config)
    issac.data_collection()

    #issac.simulate()
