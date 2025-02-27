import sys
import torch
import numpy as np
import cv2
import open3d as o3d
import pytorch3d.transforms
from typing import List
from isaacgym.torch_utils import quat_mul, quat_conjugate

sys.path.append('/home/hcis-s17/multimodal_manipulation/scoop_env/')
from src.affordance import Affordance_agent
# from src.vild.utils import get_vild_prob

def get_prob(rgb_img_path, prompt_list, params, image_root):
    pass


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

class Affordance_agent_ours(Affordance_agent):
    def __init__(self, init_object_list, action_list, DH_params, joint_limit, base_pose, device):
        super().__init__(init_object_list, action_list)
        self.image_root = '/home/hcis-s17/multimodal_manipulation/scoop_env/src/affordance/ours/image'
        self.pos_offset = 0.01
        self.axis_offset = 0.03
        self.w_offset = 0.03
        self.DH_params = DH_params
        self.joint_limit = joint_limit
        self.device = device
        self.base_pose = base_pose
        self.affordance_info = None
    
    def get_affordance_info(self):
        return self.affordance_info
    
    def get_affordance(
        self, 
        rgb_img_path,
        gray_scale_img, 
        action_seq, 
        traj_dict,
        K,
        extrinsic,
        cur_pose,
        cur_joint,
        action_candidate=[], 
    ):
        # self.trajectory_clear(None, cur_pose, traj_dict['scoop'], [rgb_img_path], [gray_scale_img], K, [extrinsic])
        
        affordance = {action: 0 for action in action_candidate}
        spoon_on_hand = self.spoon_on_hand(action_seq)
        food_on_hand = spoon_on_hand and self.food_on_hand(action_seq)
        dumbwaiter_opened = self.dumbwaiter_opened(action_seq)
        
        self.affordance_info = ''
        for action in affordance.keys():
            print(action)
            state_affordable, info = self.state_affordable(action, spoon_on_hand, food_on_hand, dumbwaiter_opened)
            if not state_affordable:
                self.affordance_info += f'{action}: {info}'
                continue
            if traj_dict[action] is not None:
                joint_affordable = np.random.choice([0, 1]) # self.joint_affordable(cur_pose.clone(), cur_joint.clone(), traj_dict[action])
                if not joint_affordable:
                    self.affordance_info += f'{action}: Cannot reach the target pose'
                    continue
                traj_clear = np.random.choice([0, 1]) # self.trajectory_clear(action, cur_pose.clone(), traj_dict[action], [rgb_img_path], [gray_scale_img], K, [extrinsic])
                if not traj_clear:
                    self.affordance_info += f'{action}: Collision detected'
                    continue
            affordance[action] = 1

        return affordance
    
    def state_affordable(self, action, spoon_on_hand, food_on_hand, dumbwaiter_opened):
        if action == 'grasp_spoon':
            if spoon_on_hand:
                return False, "Cannot grasp spoon when spoon is already on hand"
        elif action == 'put_spoon_back':
            if not spoon_on_hand:
                return False, "Cannot put spoon back when spoon is not on hand"
        elif action == 'scoop':
            if not spoon_on_hand:
                return False, "Cannot scoop when spoon is not on hand"
            if food_on_hand:
                return False, "Cannot scoop when food is already in the spoon"
        elif action == 'drop_food':
            if not food_on_hand:
                return False, "Cannot drop food when food is not in the sponn"
        elif action == 'open_dumbwaiter':
            if dumbwaiter_opened:
                return False, "Cannot open dumbwaiter when it is already opened"
        elif action == 'close_dumbwaiter':
            if not dumbwaiter_opened:
                return False, "Cannot close dumbwaiter when it is already closed"
        return True, None
            
    def spoon_on_hand(self, action_seq):
        """assume the robot has no spoon at the beginning"""
        grasped = False
        for action in action_seq:
            if action == 'grasp_spoon':
                grasped = True
            elif action == 'put_spoon_back':
                grasped = False
        return grasped

    def food_on_hand(self, action_seq):
        """assume the robot has no food at the beginning"""
        food = False
        for action in action_seq:
            if action == 'scoop':
                food = True
            elif action == 'drop_food':
                food = False
        return food
    
    def dumbwaiter_opened(self, action_seq): 
        """assume the dumbwaiter is closed at the beginning"""
        opened = False
        for action in action_seq:
            if action == 'open_dumbwaiter':
                opened = True
            elif action == 'close_dumbwaiter': 
                opened = False
        return opened
            
    def spoon_on_hand_prob(self, rgb_img_path, tool_list) -> bool:
        """Use ViLD to detect the tool on hand

        Args:
            rgb_img_path (str): path to current observation image
            tool_list (List[str]): list of tool names
        """
        nms_threshold = 0.6
        min_rpn_score_thresh = 0.9
        min_box_area = 220
        max_box_num = 10
        params = nms_threshold, min_rpn_score_thresh, min_box_area, max_box_num
        prompt2tool = {f"robot's gripper with {tool}": tool for tool in tool_list}
        # prompt2tool = {f"{tool}": tool for tool in tool_list}
        probs = get_prob(rgb_img_path, list(prompt2tool.keys()), params, image_root=self.image_root)
        exists = {prompt2tool[prompt]: prob > 0 for prompt, prob in probs.items()}
        return exists
    
    def empty_hand_prob(self, rgb_img_path) -> bool:
        """Use ViLD to detect if the hand is empty

        Args:
            rgb_img_path (str): path to current observation image
        """
        nms_threshold = 0.6
        min_rpn_score_thresh = 0.9
        min_box_area = 220
        max_box_num = 10
        prompt = "robot's gripper with nothing"
        params = nms_threshold, min_rpn_score_thresh, min_box_area, max_box_num
        probs = get_prob(rgb_img_path, [prompt], params, image_root=self.image_root)
        return probs[prompt] > 0
    
    def fk_solver(self, q: torch.Tensor):
        # TODO
        # print(self.base_pose) # checked it is the pose of the base in the world frame
        A = pose7d_to_matrix(self.base_pose) # Shape (1, 4, 4)
        j_eef = torch.zeros((1, 7, 6), device=self.device)  # Shape (1, 6, 7)
        T_list = [A]
        for i in range(len(self.DH_params)):
            a, d, alpha = list(self.DH_params[i].values())
            theta = q[:, i]
            a = torch.tensor(a, device=self.device, dtype=torch.float32)
            d = torch.tensor(d, device=self.device, dtype=torch.float32)
            alpha = torch.tensor(alpha, device=self.device, dtype=torch.float32)
            # T = torch.tensor([
            #     [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            #     [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            #     [0, np.sin(alpha), np.cos(alpha), d],
            #     [0, 0, 0, 1]
            # ], device=self.device, dtype=torch.float32)
            # T = torch.tensor([
            #     [np.cos(theta), -np.sin(theta), 0, a],
            #     [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
            #     [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
            #     [0, 0, 0, 1]
            # ], device=self.device, dtype=torch.float32)
            T = torch.tensor([
                [torch.cos(theta), -torch.sin(theta), 0, a],
                [torch.sin(theta) * torch.cos(alpha), torch.cos(theta) * torch.cos(alpha), -torch.sin(alpha), -d * torch.sin(alpha)],
                [torch.sin(theta) * torch.sin(alpha), torch.cos(theta) * torch.sin(alpha), torch.cos(alpha), d * torch.cos(alpha)],
                [0, 0, 0, 1]
            ], device=self.device, dtype=torch.float32)
            T_list.append(T_list[-1] @ T)
            
        O_end = T_list[-1][:, :3, 3]
        for i, T in enumerate(T_list[:-1]):
            Z = T[:, :3, 2]
            O = T[:, :3, 3]
            j_eef[:, i] = torch.cat((torch.cross(Z, (O_end - O)), Z), dim=-1)
        A = T_list[-1]
        T_last = torch.tensor([
            [0.7071068, 0.7071068, 0, 0],
            [-0.7071068, 0.7071068, 0, 0],
            [0, 0, 1, 0.107],
            [0, 0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        pose_7d = matrix_to_pose_7d(A @ T_last)
        
        return pose_7d, j_eef.transpose(1, 2)

    def ik_solver(self, dpose, j_eef, damping=0.2):
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(j_eef.shape[0], 7)
        return u

    def numerical_jacobian(self, q: torch.Tensor, delta=1e-6):
        """
        Compute Jacobian using finite differences.

        Args:
            q (torch.Tensor): Joint angles tensor (1, N)
            delta (float): Small step for numerical differentiation

        Returns:
            torch.Tensor: Jacobian matrix (6, N)
        """
        q = q.clone().detach()
        J = torch.zeros((6, len(q[0])), device=q.device)
        
        pose_0 = self.fk_solver(q)[0].squeeze()  # FK at q
        pos_0 = pose_0[:3]  # Extract position (x, y, z)
        quat_0 = pose_0[3:]  # Extract orientation (qw, qx, qy, qz)
        
        for i in range(len(q[0])):
            q_perturbed = q.clone()
            q_perturbed[:, i] += delta
            pose_perturbed = self.fk_solver(q_perturbed)[0].squeeze()
            
            pos_perturbed = pose_perturbed[:3]  # Extract new position
            quat_perturbed = pose_perturbed[3:]  # Extract new orientation

            # Compute numerical derivative for position
            J[:3, i] = (pos_perturbed - pos_0) / delta

            # Compute numerical derivative for orientation (convert quaternion difference to angular velocity)
            quat_diff = quat_perturbed - quat_0
            omega = 2 * quat_diff / delta  # Approximate angular velocity
            J[3:, i] = omega[1:]  # Extract only (qx, qy, qz) components
                
        return J
    
    def joint_affordable(self, cur_pose, cur_joint, traj, delta=1, step_num=1, step_size=1 / 60.):
        ## TODO 
        ## 1. get fix trajectory based on action (might to do refactoring)
        ## 2. calculate each step of joint pose based on ik solver 
        ## 3. check if each step is affordable (check robot joint limit and poses in each step)
        """
        Args:
            cur_pose: torch.tensor, shape=(7,), current pose of the robot
            traj: List[torch.tensor], list of target poses
            cur_joint: torch.tensor, shape=(7,), current joint pose of the robot
            joint_limit: torch.tensor, shape=(7, 2), joint limit of the robot
            j_eef: torch.tensor, shape=(7, 6), jacobian matrix of the end effector
        """
        
        def check_joint_limit(joint_pose):
            return torch.all(joint_pose > self.joint_limit[:, 0]) and torch.all(joint_pose < self.joint_limit[:, 1])
        
        def orientation_error(desired, current):
            cc = quat_conjugate(current)
            q_r = quat_mul(desired, cc)
            return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
        
        
        _, j_eef = self.fk_solver(cur_joint)
        # print('FK jeef', j_eef)
        # print('J', self.numerical_jacobian(cur_joint))
        # j_eef = self.numerical_jacobian(cur_joint).unsqueeze(0)
        
        cnt = 1
        while len(traj) > 0:
            pose = traj[0]
            diff_pos = torch.norm(pose[:, :3] - cur_pose[:, :3])
            diff_axis, diff_w = self._calculate_quat_diff(pose[:, 3:], cur_pose[:, 3:])
            if True: # diff_pos < self.pos_offset and diff_axis < self.axis_offset and diff_w < self.w_offset:
                traj.pop(0)
                print(cnt)
                cnt += 1
                pass
            print('Joint', cur_joint)
            pos_err = torch.where(diff_pos > self.pos_offset, pose[:, :3] - cur_pose[:, :3], torch.tensor([0., 0., 0.], device=self.device))
            orn_err = torch.where(diff_axis > self.axis_offset or diff_w > self.w_offset, orientation_error(pose[:, 3:], cur_pose[:, 3:]), torch.tensor([0., 0., 0.], device=self.device))
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1) * delta
            djoint = self.ik_solver(dpose, j_eef)
            cur_joint += djoint
            cur_pose, j_eef = self.fk_solver(cur_joint)
            print('Pose', pose)
            print('Cur pose', cur_pose)
            # for _ in range(step_num):
            #     if not check_joint_limit(cur_joint):
            #         return False
            #     cur_joint += djoint * step_size
            #     cur_pose, j_eef = self.fk_solver(cur_joint)
            #     j_eef = self.numerical_jacobian(cur_joint).unsqueeze(0)
            
            # x_ee, q_ee = cur_pose[:, :3], cur_pose[:, 3:]
            # v = (j_eef @ djoint.unsqueeze(-1)).squeeze(-1)
            # # Update position
            # x_ee += v[..., :3] * step_size
            # # Update orientation using quaternion derivative
            # omega = v[..., 3:6]  # Angular velocity (3D)
            # omega_quat = torch.cat((omega, torch.tensor([[0]], device=self.device)), dim=-1)  # Convert to quaternion (x,y,z,0)
            # q_ee = q_ee + 0.5 * step_size * quat_mul(q_ee, omega_quat)
            # q_ee = q_ee / torch.norm(q_ee)  # Normalize quaternion
            # cur_pose = torch.cat((x_ee, q_ee), -1)
        return True
    
    def trajectory_clear(self, action, cur_pose, traj, rgb_path_list, depth_path_list, K, extrinsic_list):
        ## TODO
        ## 1. get fix trajectory based on action
        ## 2. build a point cloud
        ## 3. point motion planning in 3d and detect collision
        for rgb_path, depth_path, extrinsic in zip(rgb_path_list, depth_path_list, extrinsic_list):
            
            rgb = cv2.imread(rgb_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            K = K.cpu().numpy() if isinstance(K, torch.Tensor) else K
            extrinsic = extrinsic.cpu().numpy() if isinstance(extrinsic, torch.Tensor) else extrinsic
            points, colors = self.rgbd_to_point_cloud(rgb, depth, K, depth_scale=1000.)
            _points = points[:, (0, 2, 1)] # [y, z, x] -> [y, x, z]
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(_points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=point_cloud.get_center())
            o3d.visualization.draw_geometries([point_cloud, axis])
            print('Min x:', _points[:, 0].min())
            print('Max x:', _points[:, 0].max())
            print('Min y:', _points[:, 1].min())
            print('Max y:', _points[:, 1].max())
            print('Min z:', _points[:, 2].min())
            print('Max z:', _points[:, 2].max())
            
            # extrinsic = np.linalg.inv(extrinsic)
            # print('Extrinsic\n', extrinsic)
            # points = self.transform_to_world(points[:, (2, 0, 1)], extrinsic)
            # points = points[:, (1, 0, 2)] # [x, y, z] -> [y, x, z]
            
            traj = [cur_pose] + traj
            print('Traj')
            print(traj)
            for pose in traj:
                pose_homogeneous = np.concatenate((pose[:, :3], np.array([[1]])), axis=1)
                pose = (extrinsic @ pose_homogeneous.T).T
                points = np.append(points, pose[:, (1, 2, 0)], axis=0)
                colors = np.append(colors, np.array([[0, 0, 0]]), axis=0)
            # points = points[:, (0, 2, 1)]
            print('Min x:', points[:, 0].min())
            print('Max x:', points[:, 0].max())
            print('Min y:', points[:, 1].min())
            print('Max y:', points[:, 1].max())
            print('Min z:', points[:, 2].min())
            print('Max z:', points[:, 2].max())
            
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=point_cloud.get_center())
            o3d.visualization.draw_geometries([point_cloud, axis])
            
    
    def transform_to_world(self, points, extrinsic):
        # TODO
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        points_world_homogeneous = (extrinsic @ points_homogeneous.T).T
        # points_world_homogeneous = points_homogeneous @ extrinsic 
        points_world = points_world_homogeneous[:, :3]

        return points_world
    
    def rgbd_to_point_cloud(self, rgb, depth, K, depth_scale=1.):
        h, w = depth.shape
    
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32) / depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        b = rgb[:, :, 0].reshape(-1) / 255.0
        g = rgb[:, :, 1].reshape(-1) / 255.0
        r = rgb[:, :, 2].reshape(-1) / 255.0
        colors = np.stack((r, g, b), axis=-1)

        return points, colors 

    def nor_pcd(self, points):

        seg_info = points[:, 3].reshape(len(points), 1)
        points = points[:, :3]

        # normalize the pcd
        
        centroid = np.mean(points, axis=0)
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))


        centroid = [ 0.59115381, -0.1113387 ,  0.0755547 ]
        m = 0.6797932094342392
        
        points = points - centroid
        points = points / m

        seg_pcd = np.concatenate((points, seg_info), axis=1)
        return seg_pcd
    
    def align_point_cloud(self, points, target_points=10000):
        num_points = len(points)
    
        if num_points >= target_points:
            # Randomly downsample to target_points
            indices = np.random.choice(num_points, target_points, replace=False)
            indices = np.sort(indices)

        else:
            # Resample with replacement to reach target_points
            indices = np.random.choice(num_points, target_points, replace=True)
            indices = np.sort(indices)

        new_pcd = np.asarray(points)[indices]
        
        return new_pcd

    def _check_pcd_color(self, pcd, sim_coord_data, real_coord_data):

        color_map = {
            0: [1, 0, 0],    # Red
            4: [0, 1, 0],    # Green
            1: [0, 0, 1],    # Blue
            3: [1, 1, 0],    # Yellow
            5: [1, 0, 1],     # Magenta
            2: [1, 0.5, 0]
        }
        points = []
        colors = []
    
        
        for i in range(pcd.shape[0]):
            points.append(pcd[i][:3])
            if pcd.shape[1] == 4:
                colors.append(color_map[pcd[i][3]])


        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([point_cloud])
        
    
    def _calculate_quat_diff(self, desired_quat, cur_quat):
        to_axis = desired_quat[:, :3] - cur_quat[:, :3]
        axis_dist = torch.norm(to_axis)
        w_dist = desired_quat[:, -1] - cur_quat[:, -1]
        w_dist = abs(w_dist)            
        return axis_dist, w_dist
        
if __name__ == '__main__':
    agent = Affordance_agent_ours(None, None)
    # rgb_img_path = '/home/hcis-s17/multimodal_manipulation/scoop_env/src/affordance/classifier/data/spoon/2/0_rgb/021.png'
    # print(agent.spoon_on_hand(rgb_img_path, ['spoon', 'fork', 'knife']))
    # print(agent.empty_hand(rgb_img_path))