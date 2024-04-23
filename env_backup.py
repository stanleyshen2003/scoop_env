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

class IsaacSim():
    def __init__(self, userDefinedSettings):
        #tool_type : spoon, knife, stir, fork
        self.tool = "stir"
        #set ball_amount
        self.ball_amount = 10
        
        # initialize gym
        self.gym = gymapi.acquire_gym()
        #self.domainInfo = WeighingDomainInfo(userDefinedSettings=userDefinedSettings, domain_range=None, flag_list=None)

        # create simulator
        self.env_spacing = 1.5
        self.max_episode_length = 195
        self.asset_root = "urdf"
        self.gravity = -1.0

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
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "move_scoop")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_U, "move_fork")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "scoop")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "stir")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "fork")
        
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "gripper_close")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "save")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "quit")

        # Look at the first env
        self.cam_pos = gymapi.Vec3(0.7, 0, 1.2)
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

    def create_sim(self):
        
        # parse arguments
        args = gymutil.parse_arguments(description="Joint control Methods Example")

        args.use_gpu = False
        args.use_gpu_pipeline = False
        self.device = 'cpu'
        #self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_envs = 1

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, self.gravity)

        #黏稠度
        sim_params.dt = 1.0/20

        sim_params.substeps = 1

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1

        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.contact_offset = 0.009
        sim_params.physx.rest_offset = 0.000001
        sim_params.physx.max_depenetration_velocity = 1000

        
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        #self.gym.prepare_sim(self.sim)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_table(self):

        # create table asset
        self.table_dims = gymapi.Vec3(0.8, 1, self.default_height)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, asset_options)
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.5, 0, 0)
        
    def create_bowl(self):
        def calculate_bowl_dist(pose1, pose2):
            return np.sqrt((pose1.p.x - pose2.p.x) ** 2 + (pose1.p.y - pose2.p.y) ** 2)
            
        self.bowl_num = 3
        self.bowl_pose = []
        self.min_bowl_dist = 0.3
        file_name = 'bowl/bowl.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 500000
        self.bowl_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        for _ in range(self.bowl_num):
            random_sample = False
            bowl_pose = gymapi.Transform()
            bowl_pose.r = gymapi.Quat(1, 0, 0, 1)
            while not random_sample:
                bowl_pose.p = gymapi.Vec3(0.2 + (0.6) * random.random(), -0.4 + (0.8) * random.random(), self.default_height/2)
                random_sample = True
                for pose in self.bowl_pose:
                    if calculate_bowl_dist(pose, bowl_pose) < self.min_bowl_dist:
                        random_sample = False
                        break
            self.bowl_pose.append(bowl_pose)
            
    def create_butter(self):
        file_name = 'food/butter.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.food_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        self.butter_poses = []
        butter_pose = gymapi.Transform()
        butter_pose.r = gymapi.Quat(1, 0, 0, 1)
        butter_pose.p = gymapi.Vec3(0.4, 0., self.default_height/2)
        self.butter_poses.append(butter_pose)
        butter_pose.p = gymapi.Vec3(0.4, 0.05, self.default_height/2)
        self.butter_poses.append(butter_pose)
        
    def add_food(self, env_ptr):
        for butter_pose in self.butter_poses:
            self.food_handle = self.gym.create_actor(env_ptr, self.food_asset, butter_pose, "food", 0, 0)
            food_idx = self.gym.get_actor_index(env_ptr, self.food_handle, gymapi.DOMAIN_SIM)
            self.butter_indices.append(food_idx)       
        
    def create_spoon(self):
        
        # Load spoon asset
        file_name = 'grab_spoon/spoon.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.spoon_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        
        self.spoon_pose = gymapi.Transform()
        self.spoon_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.spoon_pose.p = gymapi.Vec3(0.5, 0 , self.default_height/2+1)  

    def create_ball(self):
        self.ball_radius, self.ball_mass, self.ball_friction = 0.005,3e-05 ,1.0
        self.between_ball_space = 0.04
        ballGenerator = BallGenerator()
        file_name = 'BallHLS.urdf'
        ballGenerator.generate(file_name=file_name, ball_radius=self.ball_radius, ball_mass=self.ball_mass)
        self.ball_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, gymapi.AssetOptions())
    
    def set_ball_property(self, env_ptr, ball_pose):

        ball_friction = self.ball_friction
        ball_restitution = 0
        ball_rolling_friction = 1
        ball_torsion_friction = 1
        ball_handle = self.gym.create_actor(env_ptr, self.ball_asset, ball_pose, "grain", 0, 0)
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(env_ptr, ball_handle)

        body_shape_prop[0].friction = ball_friction
        body_shape_prop[0].rolling_friction = ball_rolling_friction
        body_shape_prop[0].torsion_friction = ball_torsion_friction
        body_shape_prop[0].restitution = ball_restitution



        self.gym.set_actor_rigid_shape_properties(env_ptr, ball_handle, body_shape_prop)
        return ball_handle
    
    def add_ball(self, env_ptr):
        #add balls
        c = np.array([115, 78, 48]) / 255.0
        color = gymapi.Vec3(c[0], c[1], c[2])

        ball_pose = gymapi.Transform()
        z = self.default_height/2 + 0.2
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)

        ball_spacing = self.between_ball_space
        
        
        for bowl_pose in self.bowl_pose:
            ball_amount = self.ball_amount
            ran = min(ball_amount, 8)
            while ball_amount > 0:
                y = bowl_pose.p.y - 0.025
                for j in range(ran):
                    x = bowl_pose.p.x - 0.02
                    for k in range(ran):
                        ball_pose.p = gymapi.Vec3(x, y, z)
                        ball_handle = self.set_ball_property(env_ptr, ball_pose)
                        self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                        self.gym.set_actor_scale(env_ptr, ball_handle, 0.5)
                        x += ball_spacing*0.18
                    y += ball_spacing*0.18
                z += ball_spacing * 0.2
                ball_amount -= 1

    def create_franka(self):
        # create franka asset
        self.num_dofs = 0
        asset_file_franka = f"franka_description/robots/{self.tool}_franka.urdf"
        

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
        self.franka_start_pose.p = gymapi.Vec3(0, 0.0, 0.2)
        self.franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    def add_franka(self, i, env_ptr):
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

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(spacing, 0.75 * spacing, spacing)
        self.default_height = 0.9
        self.create_table()
        self.create_bowl()
        self.create_ball()
        self.create_franka()
        self.create_spoon()
        self.create_butter()
    
        # cache some common handles for later use
        self.camera_handles = []
        self.franka_indices, self.kit_indices, self.urdf_indices = [], [], []
        self.franka_dof_indices = []
        self.franka_hand_indices = []
        self.urdf_link_indices = []
        self.bowl_indices = []
        self.butter_indices = []
        self.envs = []

        # create and populate the environments
        for i in range(num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            #add bowl
            
            for pose in self.bowl_pose:
                bowl_handle = self.gym.create_actor(env_ptr, self.bowl_asset, pose, "bowl", 0, 0)
                self.gym.set_actor_scale(env_ptr, bowl_handle, 0.5)
                bowl_idx = self.gym.get_actor_rigid_body_index(env_ptr, bowl_handle, 0, gymapi.DOMAIN_SIM)
                self.bowl_indices.append(bowl_idx)
            
            
            # add tabel
            self.tabel = self.gym.create_actor(env_ptr, self.table_asset, self.table_pose, "table", 0, 0)
            
            # add food
            self.add_food(env_ptr)
            self.add_ball(env_ptr)
            # add franka
            self.add_franka(i, env_ptr)

            #add spoon
            #self.spoon = self.gym.create_actor(env_ptr, self.spoon_asset, self.spoon_pose, "spoon", 0, 0)


        self.bowl_indices = to_torch(self.bowl_indices, dtype=torch.long, device=self.device)
        self.franka_indices = to_torch(self.franka_indices, dtype=torch.long, device=self.device)
        self.franka_dof_indices = to_torch(self.franka_dof_indices, dtype=torch.long, device=self.device)
        self.franka_hand_indices = to_torch(self.franka_hand_indices, dtype=torch.long, device=self.device)
        self.urdf_indices = to_torch(self.urdf_indices, dtype=torch.long, device=self.device)
        self.urdf_link_indices = to_torch(self.urdf_link_indices, dtype=torch.long, device=self.device)
        self.kit_indices = to_torch(self.kit_indices, dtype=torch.long, device=self.device)
        self.butter_indices = to_torch(self.butter_indices, dtype=torch.long, device=self.device)
        

    def reset(self):
        self.franka_init_pose = torch.tensor([-0.4969, -0.5425,  0.3321, -2.0888,  0.0806,  1.6983,  0.5075,  0.0400, 0.0400], dtype=torch.float32, device=self.device)
        
        self.dof_state[:, self.franka_dof_indices, 0] = self.franka_init_pose
        self.dof_state[:, self.franka_dof_indices, 1] = 0
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
        self.scoop_state_reset()
        self.is_stirring = False
        self.is_moving = False
    
    def scoop_state_reset(self):
        self.is_scooping = False
        self.scoop_up = False
        self.scoop_delta = 0.3
        

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
    
    def move(self, pos, rot=torch.Tensor([[1., 0., 0., 0.]])):
        zero = torch.Tensor([0, 0, 0])
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        goal_offset = 0.01
        xy_offset = 0.03
        w_offset = 0.05
        axis_offset = 0.08
        if not self.is_moving:
            self.is_moving = True
        to_goal = pos - hand_pos
        xy_dist = torch.norm(to_goal[:, :2], dim=1)
        goal_dist = torch.norm(to_goal, dim=1)
        to_axis = rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1)
        w_dist = abs(rot[:, -1] - hand_rot[:, -1])
        pos_err = torch.where(goal_dist > goal_offset, pos - hand_pos, zero)
        pos_err = torch.where(xy_dist > xy_offset, torch.cat([pos_err[:, :2], torch.Tensor([[0]])], -1), pos_err)
        orn_err = torch.where(axis_dist > axis_offset or w_dist > w_offset, self.orientation_error(rot, hand_rot), zero)
        if goal_dist <= goal_offset and axis_dist <= axis_offset and w_dist <= w_offset:
            self.is_moving = False
            print("finish moving")
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)        
        return dpose
        
        
    def scoop(self):
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        goal_offset = 0.02
        w_offset = 0.05
        axis_offset = 0.08

        if not self.is_scooping:
            print("start scooping")
            # goal_pos, _ = torch.sort(self.rb_state_tensor[self.ball_handle_list, :3], dim=0)
            # goal_pos, _ = goal_pos.median(dim=0)
            
            goal_pos = hand_pos + torch.Tensor([[0.0, -0.08, -0.07]])
            self.scoop_goal_pos = goal_pos
            self.scoop_goal_rot = torch.Tensor([[0.7, 0.6, 0.5, 0.0]]).to(self.device)
            self.is_scooping = True
            
        
        to_goal = self.scoop_goal_pos - hand_pos
        goal_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        to_axis = self.scoop_goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = self.scoop_goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(goal_dist > goal_offset, self.scoop_goal_pos - hand_pos, torch.Tensor([0, 0, 0]))
        orn_err = torch.where(axis_dist > axis_offset or w_dist > w_offset, self.orientation_error(self.scoop_goal_rot, hand_rot), torch.Tensor([0, 0, 0]))
        
        if self.is_scooping and not self.scoop_up and (goal_dist <= goal_offset and axis_dist <= axis_offset and w_dist <= w_offset):
            print("scooping up")
            self.scoop_delta = 0.05
            self.scoop_goal_pos[:, 2] += 0.1
            self.scoop_up = True
        elif self.scoop_up and goal_dist <= goal_offset:
            print("scooping finish")
            self.scoop_state_reset()
        
        
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)        
        return dpose
    
    def stir(self):
        
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        
        goal_offset = 0.01
        w_offset = 0.05
        axis_offset = 0.08
        
        if not self.is_stirring:
            print("start stirring")
            self.is_stirring = True
            self.stir_delta = 0.02
            self.stir_center = hand_pos - torch.Tensor([[-0.05, 0, 0.3]])
        bowl_dir = self.stir_center - hand_pos
        bowl_dist = torch.norm(bowl_dir[:, :1], dim=1).unsqueeze(-1)
        stir_radius = 0.055
        if bowl_dist > stir_radius:
            print("reset", bowl_dir, bowl_dist)
            self.is_stirring = False
            return torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
        goal_xy = bowl_dir[:, [1, 0, 2]]
        goal_xy[:, 0] *= -1
        goal_xy[:, 2] = 0.
        goal_norm = torch.norm(goal_xy, dim=1).unsqueeze(-1)
        goal_xy /= goal_norm
        pos_err = goal_xy
        orn_err = torch.Tensor([[0, 0, 0]])
        
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)        
        return dpose
        
    def fork(self):
        hand_pos = self.rb_state_tensor[self.franka_hand_indices, :3]
        hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
        
        
        goal_offset = 0.01
        w_offset = 0.05
        axis_offset = 0.08
        
        self.fork_delta = 0.3
        dpose = torch.Tensor([[[0.],[0.],[-1.],[0.],[0.],[0.]]])
        
        return dpose
        
        
        
        
        
    def data_collection(self):
        self.reset()

        action = ""

        while not self.gym.query_viewer_has_closed(self.viewer):
            bowl_idx = 0
            bowl_pos = self.rb_state_tensor[self.bowl_indices[bowl_idx], :3]
            critical_pos = bowl_pos + torch.Tensor([[-0.05, 0, 0.3]])
            critical_rot = torch.Tensor([[1.0, 0.25, 0., 0.]])
            fork_food_pos = self.rb_state_tensor[self.butter_indices, :3]
            fork_food_pos[:, -1] += 0.3
            # critical_rot = quat_from_angle_axis(torch.Tensor([0.3]), to_bowl)
            # critical_rot = torch.cat([to_bowl.view(3, ), torch.Tensor([0.8])], -1).unsqueeze(-1).view(1, 4)
            
            # ciritical_yaw_dir = self.quat_axis(critical_rot, 0)
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
            moving_delta = 0.3
            print_pos = True
            


            
            for evt in self.gym.query_viewer_action_events(self.viewer):
                action = evt.action if (evt.value) > 0 else ""
                
            if action == "up":
                dpose = torch.Tensor([[[0.],[0.],[1.],[0.],[0.],[0.]]]).to(self.device) * delta
            elif action == "down":
                dpose = torch.Tensor([[[0.],[0.],[-1.],[0.],[0.],[0.]]]).to(self.device) * delta
            elif action == "left":
                dpose = torch.Tensor([[[0.],[-1.],[0.],[0.],[0.],[0.]]]).to(self.device) * delta
            elif action == "right":
                dpose = torch.Tensor([[[0.],[1.],[0.],[0.],[0.],[0.]]]).to(self.device) * delta
            elif action == "backward":
                dpose = torch.Tensor([[[-1.],[0.],[0.],[0.],[0.],[0.]]]).to(self.device) * delta
            elif action == "forward":
                dpose = torch.Tensor([[[1.],[0.],[0.],[0.],[0.],[0.]]]).to(self.device) * delta
            elif action == "turn_left":
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[-10.]]]).to(self.device) * delta
            elif action == "turn_right":
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[10.]]]).to(self.device) * delta
            elif action == "turn_up":
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[10.],[0.]]]).to(self.device) * delta
            elif action == "turn_down":
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[-10.],[0.]]]).to(self.device) * delta
            elif action == "test_pos":
                dpose = torch.Tensor([[[0.],[0.],[0.],[1.],[0.],[0.]]]).to(self.device) * delta
            elif action == "test_neg":
                dpose = torch.Tensor([[[0.],[0.],[0.],[-1.],[0.],[0.]]]).to(self.device) * delta
            elif action == "scoop":
                dpose = self.scoop()
                dpose = dpose.to(self.device) * self.scoop_delta
            elif action == "stir":
                dpose = self.stir()
                dpose = dpose.to(self.device) * self.stir_delta
            elif action == "move_spoon":
                dpose = self.move(critical_pos, critical_rot) * moving_delta
            elif action == "move_fork":
                dpose = self.move(fork_food_pos) * moving_delta
            elif action == "fork":
                dpose = self.fork() * self.fork_delta
            elif action == "gripper_close":
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]).to(self.device)
                if torch.all(self.pos_action[:, 7:9] == gripper_close):
                    self.pos_action[:, 7:9] = gripper_open
                elif torch.all(self.pos_action[:, 7:9] == gripper_open):
                    self.pos_action[:, 7:9] = gripper_close
            elif action == "save":
                hand_pos = self.rb_state_tensor[self.franka_hand_indices, 0:3]
                hand_rot = self.rb_state_tensor[self.franka_hand_indices, 3:7]
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]).to(self.device)
                print(hand_pos)
                print(hand_rot)
                print()
            elif action == "quit":
                break
            else:
                print_pos = False
                dpose = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]).to(self.device)

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


if __name__ == "__main__":
    issac = IsaacSim()
    issac.data_collection()

    issac.simulate()
