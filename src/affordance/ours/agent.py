from src.affordance import Affordance_agent

class Affordance_agent_ours(Affordance_agent):
    def __init__(self, init_object_list, action_list):
        super().__init__(init_object_list, action_list)
    def get_affordance(self, rgb_img, gray_scale_img, action_seq, action_candidate=[]):
        
        return super().get_affordance(rgb_img, gray_scale_img, action_seq, action_candidate)