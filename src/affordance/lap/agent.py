import numpy as np
import sys
import cv2
sys.path.append('/home/hcis-s17/multimodal_manipulation/scoop_env')
from src.affordance import Affordance_agent
from src.semantic.utils import get_messages
from src.semantic.openai_client import call_openai_api
from src.utils import *
from src.affordance.lap.vild import get_vild_prob
    
def parse_object_list(text_path):
    file_text = ''.join(open(text_path).readlines())
    list_start = file_text.find('object list: [') + len('object list: [')
    list_end = file_text.find(']', list_start)
    list_string = file_text[list_start: list_end]
    return list_string.replace("'", '').split(', ')

def parse_action_list(text_path):
    file_text = ''.join(open(text_path).readlines())
    list_start = file_text.find('Action list: [') + len('Action list: [')
    list_end = file_text.find(']', list_start)
    list_string = file_text[list_start: list_end]
    return [action[3:] for action in list_string.replace("'", '').split(', ')]

def parse_action_name(action_choise, action_list):
    return action_list[ord(action_choise) - ord('A')]

class Affordance_agent_LAP(Affordance_agent):
    def __init__(self, init_object_list, action_list):
        super().__init__(init_object_list, action_list)
        self.important_consideration = ['1. distance for action', '2. obstacle for action']
    
    def get_prompt_template(self, object_list, action, answer=None):
        return f"""Question: There are {object_list} in the environment. Please consider {''.join(self.important_consideration)}, \\
and answer me True or False if the action {action} is affordable in the environment. \\
Answer: {answer if answer else ''}"""
            
    
    def get_system_prompt(self):
        return "You should seriously consider if the action is affordable in the environment, and give True or False answer."

    def get_user_prompt(self, action):
        return self.get_prompt_template(self.init_object_list, action)
    
    def get_action_object(self, action, mode):
        action_object = []
        if mode == 'perception':
            if "move" in action:
                action_object.append(action.replace('move_to_', '').replace('_', ' '))
            elif "take_tool" in action:
                action_object.append(action.replace("take_tool ", '').replace('(', '').replace(')', ''))
            elif "scoop" in action:
                action_object.extend(['bowl', 'spoon'])
        elif mode == 'context':
            if "move" in action:
                action_object.append(action.replace('move_to_', ''))
            elif "take_tool" in action:
                action_object.append(action.replace("take_tool ", '').replace('(', '').replace(')', ''))
            
        return action_object
    
    def get_context_affordance(self, action_object=[]):
        return 1
        # since in our setting all action are available only when the related objects are in the list
        if not action_object:
            return 1
        for o in action_object:
            if not o in self.init_object_list:
                return 0
        return 1

    def get_perception_affordance(self, rgb_img, action_object=[]):
        if not action_object:
            return 1
        nms_threshold = 0.6 #@param {type:"slider", min:0, max:0.9, step:0.05}
        min_rpn_score_thresh = 0.9  #@param {type:"slider", min:0, max:1, step:0.01}
        min_box_area = 220 #@param {type:"slider", min:0, max:10000, step:1.0}
        params = nms_threshold, min_rpn_score_thresh, min_box_area
        probs = get_vild_prob(rgb_img, action_object, params)
        return probs

    def get_prompt_affordance(self, rgb_img, action, model='gpt-4o'):
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(action)
        base64_image = encode_image(rgb_img)
        messages = get_messages(system_prompt, user_prompt, base64_image)
        top_logprobs = call_openai_api(messages, model).choices[0].logprobs.content[0].top_logprobs
        top_logprobs = {top_logprob.token: top_logprob.logprob for top_logprob in top_logprobs}
        true_prob = np.exp(top_logprobs.get('True', float('-inf')))
        false_prob = np.exp(top_logprobs.get('False', float('-inf')))
        return 0.5 * int(true_prob == false_prob) + int(true_prob > false_prob)
    
    def get_affordance(self, rgb_img, gray_scale_img, action_seq, action_candidate=[]):
        affordance = {}
        if not action_candidate:
            action_candidate = self.action_list
        all_object_action_perception = []
        for action in action_candidate:
            all_object_action_perception.extend(self.get_action_object(action, 'perception'))
        all_object_action_perception = set(all_object_action_perception)
        all_perception_affordance = self.get_perception_affordance(rgb_img, list(all_object_action_perception))
        
        for action in self.action_list:
            if action == 'DONE':
                affordance[action] = 0.5
            elif action not in action_candidate:
                affordance[action] = 0.
            else:
                action_object_perception = self.get_action_object(action, 'perception')
                aff_cont = self.get_context_affordance(self.get_action_object(action, 'context'))
                aff_per = np.mean([all_perception_affordance.get(o, 0) for o in action_object_perception]) if action_object_perception else 1
                aff_prom = self.get_prompt_affordance(rgb_img, action)
                affordance[action] = aff_cont * aff_per * aff_prom
        return affordance
    
if __name__ == '__main__':
    action_list = parse_action_list('/home/hcis-s17/multimodal_manipulation/scoop_env/confidence_calibration/question/text/user/0002.txt')
    object_list = parse_object_list('/home/hcis-s17/multimodal_manipulation/scoop_env/confidence_calibration/question/text/user/0002.txt')
    action_name = parse_action_name('A', action_list)
    agent = Affordance_agent_LAP(object_list, action_list)
    print(action_list, object_list, action_name)
    print(agent.get_affordance('/home/hcis-s17/multimodal_manipulation/scoop_env/confidence_calibration/question/image/0002.jpg', None, None, None))