import os
import socket
import json
from typing import List

import sys
sys.path.append('/home/hcis-s17/multimodal_manipulation/scoop_env/')
from src.utils import encode_image, decode_image

def preprocess_action(action):
    """
    Remove the parentheses and underscores in the action string
    """
    return action.replace('(', '').replace(')', '').replace('_', ' ')

def preprocess_object(object):
    return object.replace('_', ' ')

def format_action_choices(action_list: List[str]):
    """
    Create a dictionary that maps action to a character
    """
    return {action: chr(ord('A') + i)  for i, action in enumerate(action_list)}

def segmentation_process(rgb_img_path):
    """
    send image to localhost to process image with segmentation model
    """
    host = '127.0.0.1'
    port = 12345
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        data = {'img_path': rgb_img_path}
        s.sendall(json.dumps(data).encode('utf-8'))
        response = s.recv(4096)
        response = response.decode('utf-8')
    if os.path.exists(response):
        return response
    return None
    
def generate_prompt(action_seq, indent=False):
    ret = ""
    for i, action in enumerate(action_seq):
        executed_action = " ".join([f'{j+1}. {action_seq[j]}' for j in range(i)])
        if indent:
            ret += f"""
    Iteration {i+1}:
        Output: {action}"""
        else:
            ret += f"""Iteration {i+1}:
    Output: {action}
"""
    # Input: {executed_action}
    return ret

def get_action_description_prompt():
    action_descrption = {
        'take_tool (tool name)': 'take the tool from the tool holder, there must be empty of the robotics hand', 
        'put_tool (tool_name)': 'put the tool back to the tool holder.', 
        'move_to_container': 'move to the container for further action like pulling or scooping', 
        'scoop': 'scoop the food, the speed of scooping will be affected by food state.', 
        'stir': 'stir the food.', 
        'put_food': 'put the food on your tool into the container. Note that you should move to the destination container before putting the food.', 
        'pull_bowl_closer': 'pull the nearest bowl to the center of the table', 
        'DONE': 'indicates that the instruction is done.'
    }
    
    action_description_prompt = '\n'.join([f'{i + 1}. {action_name}: {action_descrption[action_name]}' for i, action_name in enumerate(action_descrption)])
    return action_description_prompt

def get_example_prompt(use_vlm=False, selection=False):
    system_prompt = ""
    system_image_url = []
    example_path = '/home/hcis-s17/multimodal_manipulation/scoop_env/src/semantic/example/text'
    example_id = 1
    for txt in os.listdir(example_path):
        if not txt.endswith('.txt'):
            continue
        txt_file = os.path.join(example_path, txt)
        img_file = txt_file.replace('txt', 'jpg').replace('text', 'image')
        content = ''.join(open(txt_file).readlines()).split('\n\n')
        example_instruction = content[0]
        example_action_seq = content[1].split('\n')
        example_container_list = content[2].split('\n')
        example_action_list = content[3].split('\n')
        if len(content) > 4:
            example_environment = content[4].replace('\n', '\n        ')
        
        example_action_seq = [preprocess_action(action) for action in example_action_seq]
        example_action_list = [preprocess_action(action) for action in example_action_list]
        example_container_list = [preprocess_object(container) for container in example_container_list]
        example_action_dict = format_action_choices(example_action_list)
        # print(example_action_dict)
        if selection:
            # example_action_seq = [example_action_dict[action] for action in example_action_seq] only character
            example_action_seq = [f"{example_action_dict[action]}. {action}" for action in example_action_seq]
            example_action_list = [f'{selection}. {action}' for action, selection in example_action_dict.items()]
        system_prompt += f"""Example {example_id}:
    Action list: {example_action_list}
    Initial object list: {example_container_list}
    Instruction: {example_instruction}
    {generate_prompt(example_action_seq, indent=True)}
    
    Explanation:
        {example_environment}

"""
        if use_vlm and os.path.exists(img_file):
            system_image_url.append(encode_image(img_file))
        example_id += 1
    return system_prompt, system_image_url

def get_system_prompt(use_vlm=False, selection=False, with_example=True):
    system_image_url = []
    vlm_prompt = """
You should consider the information from the input image and decide the appropriate action. For example, if there are only a few beans in the bowl, making it unsuitable for scooping, avoid scooping from that bowl.
You should avoid scooping beans from the bowl too far away, or pull it closer before you scoop it, since it may cause to failed.
If the primitive might cause to collision of failure, you may pull bowl to avoid it.
"""
    system_prompt = f"""You are a robot arm in food manipulation scenario. You should focus on your gripper. You need to pick an action from the action list to finish the whole task step by step.
Please also take the previous actions into consideration when choosing the next action.
{use_vlm * vlm_prompt}
{get_action_description_prompt()}"""
    if with_example:
        example_system_prompt, system_image_url = get_example_prompt(use_vlm=use_vlm, selection=selection)
        system_prompt += example_system_prompt
    return system_prompt, system_image_url

def get_user_prompt(instruction, action_seq, action_dict, container_list, additional_info, segmentation=False) -> str:
    container_list = [preprocess_object(container) for container in container_list]
    action_seq = [preprocess_action(action) for action in action_seq]
    
    action_choices = [f"{v}. {k}" for k, v in action_dict.items()]
    action_seq_choices = [f"{action_dict[action]}. {action}" for action in action_seq]
    additional_info = f"Please also consider some additional information: {additional_info}" if additional_info else ""
    segmentation_prompt = "Please focus on the segmentation result of the robot to make the decision." if segmentation else ""
    user_prompt = f"""
Action list: {action_choices}
Initial object list: {container_list}
Instruction: {instruction}
Please choose one action from the action list to execute at the next iteration and output it directly.
{additional_info}
{segmentation_prompt}
{generate_prompt(action_seq_choices)}Iteration {len(action_seq_choices)+1}:
    Output: """
    return user_prompt

def next_action_prompt(instruction, action_seq):
    system_prompt = f"""You are a smart assistant tasked with identifying what food properties or information should be take into consideration before deciding the next action for a robot about a food manipulation task. Consider all previous actions and their outcomes when deciding. Please do not decide the action and specify what you should achieve in the next step and what you should take into consideration to achieve the goal in high level. Provide your answer in 50 words or fewer.
{get_action_description_prompt()}"""
    action_sequence = ", ".join([f"{i+1}. {action}" for i, action in enumerate(action_seq)])
    user_prompt = f"""Instruction: {instruction}
Previous actions:
{action_sequence}"""
    return system_prompt, user_prompt


def extract_from_choice_prompt(instruction, action_seq, choices, container_list):
    system_prompt = f"""You are a smart assistant tasked with identifying what food properties (extrinsic, amount, distribution) or information that should be take into consideration in a food manipulation task. You will be given some candidate actions, please think through all the choices as thorough as possible and list what I should know in order to make the choice. Please do not choose the action directly.
{get_action_description_prompt()}"""
    container_list = [preprocess_object(container) for container in container_list]
    action_seq = [preprocess_action(action) for action in action_seq]
    
    action_sequence = ", ".join([f"{i+1}. {action}" for i, action in enumerate(action_seq)])
    user_prompt = f"""Instruction: {instruction}
Previous actions: {action_sequence}
Object list: {', '.join(container_list)}
Possible next actions: {', '.join(choices)}"""
    return system_prompt, user_prompt

def extract_important_information_prompt(instruction, important_considerations, object_list=None):
    object_list = [preprocess_object(container) for container in object_list]
    system_prompt = f"""You are a great observer that can describe the environment in detail. Given an image of the food manipulation scenario, the overall goal, the object list, and the key considerations of determining the next move, extract the important information from the image and the instruction. Please provide your answer in 50 words or fewer."""
    user_prompt = f"""Overall goal: {instruction}
Key considerations: {important_considerations}
Object list: {', '.join(object_list)}"""
    return system_prompt, user_prompt

def extract_from_choice_prompt_selection(instruction, action_seq, action_candidate, action_dict, container_list):
    system_prompt = f"""You are a smart assistant tasked with identifying what food properties (extrinsic, amount, distribution) or information that should be take into consideration in a food manipulation task. You will be given some candidate actions, please think through all the choices as thorough as possible and list what I should know in order to make the choice. Please do not choose the action directly.
{get_action_description_prompt()}"""
    container_list = [preprocess_object(container) for container in container_list]
    action_seq = [preprocess_action(action) for action in action_seq]
    
    action_sequence = ", ".join([f"{action_dict[action]}. {action}" for action in action_seq])
    action_choices = [f"{action_dict[action]}. {action}" for action in action_candidate]
    user_prompt = f"""Instruction: {instruction}
Previous actions: {action_sequence}
Object list: {', '.join(container_list)}
Possible next actions: {', '.join(action_choices)}"""
    return system_prompt, user_prompt

def extract_important_information_prompt_selection(instruction, important_considerations, object_list=None):
    object_list = [preprocess_object(container) for container in object_list]
    system_prompt = f"""You are a great observer that can describe the environment in detail. Given an image of the food manipulation scenario, the overall goal, the object list, and the key considerations of determining the next move, extract the important information from the image and the instruction. Please provide your description with important information in 50 words or fewer. without choosing the action directly."""
    user_prompt = f"""Overall goal: {instruction}
Key considerations: {important_considerations}
Object list: {', '.join(object_list)}"""
    return system_prompt, user_prompt

def choose_from_information(instruction, important_information, action_candidate, action_dict, container_list):
    container_list = [preprocess_object(container) for container in container_list]
    action_choices = [f"{action_dict[action]}. {action}" for action in action_candidate]
    system_prompt = f"""You are a decision maker that can choose the next action based on the information extracted from the image and the instruction. Given the information extracted from the image and the instruction, choose the next action from the action list. Please only answer a single character from the action list."""
    user_prompt = f"""Overall goal: {instruction}
Important information: {important_information}
Object list: {', '.join(container_list)}
Action choices: {', '.join(action_choices)}"""
    return system_prompt, user_prompt

def get_messages(system_prompt, user_prompt, system_image_url=None, user_image_url=None):
    """system_image_url is not supported yet"""
    if system_image_url is not None:
        if isinstance(system_image_url, list):
            system_content = [{"type": "text", "text": system_prompt}]
            for url in system_image_url:
                system_content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        elif isinstance(system_image_url, str):
            system_content = [{"type": "text", "text": system_prompt}, {"type": "image_url", "image_url": {"url": system_image_url, "detail": "high"}}]
        else:
            raise ValueError("system_image_url should be a list or a string")
    else:
        system_content = [{"type": "text", "text": system_prompt}]
    if user_image_url is not None:
        if isinstance(user_image_url, list):
            user_content = [{"type": "text", "text": user_prompt}]
            for url in user_image_url:
                user_content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        elif isinstance(user_image_url, str):
            user_content = [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": user_image_url, "detail": "high"}}]
        else:
            raise ValueError("user_image_url should be a list or a string")
    else:
        user_content = [{"type": "text", "text": user_prompt}]
    messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
    ]
    return messages

if __name__ == '__main__':
    # system_prompt, system_image_url = get_system_prompt(use_vlm=True, selection=True)
    # print(system_prompt)
    # for i, url in enumerate(system_image_url):
    #     img = decode_image(url, f'test_{i + 1}.png')
    # response = segmentation_process('/home/hcis-s17/multimodal_manipulation/scoop_env/src/semantic/output.png')
    # print(response)
    print(get_example_prompt(use_vlm=True, selection=True)[0])