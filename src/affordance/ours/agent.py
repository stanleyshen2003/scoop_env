import sys
sys.path.append('/home/hcis-s17/multimodal_manipulation/scoop_env/')
from src.affordance import Affordance_agent
from src.vild.utils import get_vild_prob


class Affordance_agent_ours(Affordance_agent):
    def __init__(self, init_object_list, action_list):
        super().__init__(init_object_list, action_list)
        self.image_root = '/home/hcis-s17/multimodal_manipulation/scoop_env/src/affordance/ours/image'
        self.additional_information = None
    
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
        probs = get_vild_prob(rgb_img_path, list(prompt2tool.keys()), params, image_root=self.image_root)
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
        probs = get_vild_prob(rgb_img_path, [prompt], params, image_root=self.image_root)
        return probs[prompt] > 0
        
if __name__ == '__main__':
    agent = Affordance_agent_ours(None, None)
    rgb_img_path = '/home/hcis-s17/multimodal_manipulation/scoop_env/src/affordance/classifier/data/spoon/2/0_rgb/021.png'
    print(agent.tool_on_hand(rgb_img_path, ['spoon', 'fork', 'knife']))
    print(agent.empty_hand(rgb_img_path))