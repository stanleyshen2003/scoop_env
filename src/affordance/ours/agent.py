import sys
import torch
import open3d as o3d
from typing import List
sys.path.append('/home/hcis-s17/multimodal_manipulation/scoop_env/')
from src.affordance import Affordance_agent
# from src.vild.utils import get_vild_prob

def get_prob(rgb_img_path, prompt_list, params, image_root):
    pass

class Affordance_agent_ours(Affordance_agent):
    def __init__(self, init_object_list, action_list):
        super().__init__(init_object_list, action_list)
        self.image_root = '/home/hcis-s17/multimodal_manipulation/scoop_env/src/affordance/ours/image'
        self.additional_information = None
        self.pos_offset = 0.01
        self.axis_offset = 0.03
        self.w_offset = 0.03
    
    def get_additional_info(self):
        # TODO get additional information
        return self.additional_information
    
    def get_affordance(self, rgb_img_path, gray_scale_img, action_seq, action_candidate=[]):
        return super().get_affordance(rgb_img_path, gray_scale_img, action_seq, action_candidate)
    
    def tool_on_hand(self, rgb_img_path, tool_list) -> bool:
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
    
    def empty_hand(self, rgb_img_path) -> bool:
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
    
    def joint_affordable(self, cur_pose, traj, cur_joint, joint_limit, j_eef, delta=0.01):
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
        def ik_solver(dpose, damping=0.2):
            j_eef_T = torch.transpose(j_eef, 1, 2)
            lmbda = torch.eye(6, device=j_eef.device) * (damping ** 2)
            u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(7)
            return u
        
        def check_joint_limit(joint_pose, joint_limit):
            return torch.all(joint_pose > joint_limit[:, 0]) and torch.all(joint_pose < joint_limit[:, 1])
        
        pose = traj.pop(0)
        while len(traj) > 0:
            if not check_joint_limit(cur_joint, joint_limit):
                return False
            diff_pos = torch.norm(pose[:3] - cur_pose[:3])
            diff_axis, diff_w = self._calculate_quat_diff(pose[3:], cur_pose[3:])
            if diff_pos < self.pos_offset and diff_axis < self.axis_offset and diff_w < self.w_offset:
                pose = traj.pop(0)
                continue
            dpose = (pose - cur_pose) * delta
            cur_pose += dpose
            cur_joint += ik_solver(dpose)
        return True
    
    def trajectory_clear(self, action, cur_pose, traj, rgb_path_list, depth_path_list, intrinsic, extrinsic_list):
        ## TODO
        ## 1. get fix trajectory based on action
        ## 2. build a point cloud
        ## 3. point motion planning in 3d and detect collision
        pass
    
    def _calculate_quat_diff(self, desired_quat, cur_quat):
        ## TODO
        ## 1. calculate the difference between two quaternions
        to_axis = desired_quat[:3] - cur_quat[:3]
        axis_dist = torch.norm(to_axis)
        w_dist = desired_quat[-1] - cur_quat[-1]
        w_dist = abs(w_dist)            
        return axis_dist, w_dist
    
if __name__ == '__main__':
    agent = Affordance_agent_ours(None, None)
    # rgb_img_path = '/home/hcis-s17/multimodal_manipulation/scoop_env/src/affordance/classifier/data/spoon/2/0_rgb/021.png'
    # print(agent.tool_on_hand(rgb_img_path, ['spoon', 'fork', 'knife']))
    # print(agent.empty_hand(rgb_img_path))