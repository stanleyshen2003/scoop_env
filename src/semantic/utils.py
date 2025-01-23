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
    action_descrption_v0 = {
        'take_tool (tool name)': 'take the tool from the tool holder, there must be empty of the robotics hand', 
        'put_tool (tool_name)': 'put the tool back to the tool holder.', 
        'move_to_container': 'move to the container for further action like pulling or scooping', 
        'scoop': 'scoop the food, the speed of scooping will be affected by food state.', 
        'stir': 'stir the food.', 
        'put_food': 'put the food on your tool into the container. Note that you should move to the destination container before putting the food.', 
        'pull_bowl_closer': 'pull the nearest bowl to the center of the table', 
        'DONE': 'indicates that the instruction is done.'
    }
    action_descrption_v1 = {
        'grasp_spoon': "Grasp the spoon from the tool holder. The robot arm must have no tools in the gripper when choosing this action.",
        'put_spoon_back': "Put the spoon back to the tool holder.",
        'move_to_container': "Move to a container for actions like pulling or scooping.",
        'scoop': "Scoop food, with the speed adapted to the food's state.",
        'stir': "Stir the food.",
        'drop_food': "When the robot arm is positioned above a container, drop the food from the spoon into the container.",
        'pull_bowl_closer': "Pull the nearest bowl toward the center of the table.",
        'open_microwave': "Open the microwave door.",
        'close_microwave': "Close the microwave door.",
        'put_bowl_in_microwave': "Place the nearest bowl into the microwave.",
        'start_microwave': "Start the microwave.",
        'DONE': "Indicate that the task is complete."

    }
    
    action_description_prompt = '\n'.join([f'{i + 1}. {action_name}: {action_descrption_v1[action_name]}' for i, action_name in enumerate(action_descrption_v1)])
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
    vlm_prompt_v0 = """
You should consider the information from the input image and decide the appropriate action. For example, if there are only a few beans in the bowl, making it unsuitable for scooping, avoid scooping from that bowl.
You should avoid scooping beans from the bowl too far away, or pull it closer before you scoop it, since it may cause to failed.
If the primitive might cause to collision of failure, you may pull bowl to avoid it.
"""
    vlm_prompt_v1 = """
Analyze the input image and determine the most suitable action. Follow these guidelines to ensure appropriate decisions:

1. Scooping Actions:
Avoid scooping from bowls that contain only a few beans, as they are unsuitable for this action.
If a bowl is too far away, either avoid it or move it closer before attempting to scoop.
2. Collision Avoidance:
If the action might cause a collision or result in a failure, reposition the bowl to create a safer setup before proceeding.
Provide a clear, logical decision aligned with the conditions above. If uncertain, suggest corrective actions instead of proceeding with the task.
"""
    vlm_prompt_v2 = """
Additional Knowledges
1. Scooping Actions
Avoid scooping from bowls with insufficient food (e.g., only a few beans).
If a bowl is too far, pull it closer before attempting to scoop.
2. Collision Avoidance
If an action risks a collision or task failure, pull the bowl to a safer location before proceeding.
"""

    base_prompt_v0 = """
You are a robot arm in a food manipulation scenario. Focus on your gripper and choose an action from the list to complete the task step by step. Consider the previous actions when selecting the next action.
"""
    base_prompt_v1 = """
Scenario: You are a robotic arm designed for food manipulation tasks. Your goal is to complete the assigned task step by step by selecting the most appropriate actions from the provided action list.

Instructions:

1. Gripper Management:

Pay close attention to the state and position of your gripper during each step.
Ensure that the gripper is appropriately aligned, adjusted, and calibrated for the current action.
2. Sequential Decision-Making:

Carefully evaluate the outcomes of previous actions when choosing the next action.
Maintain consistency and avoid redundancy by considering the cumulative effects of all prior steps.
3. Action Selection:

From the action list, select the most logical and efficient action required to move closer to completing the task.
Prioritize actions that ensure precision, safety, and task progression.
4. Feedback Integration:

Continuously assess task progress and incorporate real-time feedback to adjust subsequent actions as needed.

Goal: Complete the task efficiently while maintaining accuracy, safety, and a smooth sequence of actions.
"""
    base_prompt_v2 = """
Scenario
You are a robotic arm specialized in food manipulation tasks. Your mission is to complete the assigned task step-by-step by selecting the most appropriate actions from the provided list. Your decisions should balance precision, safety, efficiency, and task progression.

Guidelines for Action Selection
1. Scenario Observation
Start by analyzing the input image or scenario details. Evaluate the environment to identify the food items, bowl positions, tool availability, and potential obstacles.
Use this observation as the foundation for selecting the appropriate next action.
2. Gripper Management
Before executing any action, check the gripper's state and position to confirm it is aligned, adjusted, and calibrated for the specific task at hand.
3. Sequential Decision-Making
Analyze the outcomes of previous steps to ensure consistency and avoid unnecessary repetition.
Focus on actions that bring the task closer to completion while considering cumulative effects.
4. Action Selection
From the provided action list, choose the most logical and efficient action required to advance the task.
Prioritize precision, safety, and progression toward the goal.
Suggest corrective actions when the current setup is unsuitable.
"""
    system_prompt = f"""{base_prompt_v2}
{use_vlm * vlm_prompt_v2}
Action Description
{get_action_description_prompt()}

Scenario Format
You will be presented with a single scenario containing the following details:

1. Action List: A list of all actions that the robot can perform, formatted as character. action.
2. Initial Object List: A detailed inventory of objects present in the environment, formatted as container_name (food inside).
3. Instruction: The high-level task or goal that the robot must accomplish.
4. Iterative Previous Actions: A chronological record of the actions the robot has executed in prior iterations.
5. Current Observation: An image of the robot's current environment.

Input Format

You will be provided with several examples, each illustrating a unique scenario in the format described above.
Following these, another scenario will be presented, requiring you to deduce and choose the next optimal action.
Important Note: In the provided examples, the image represents the observation from the initial state of the environment, not the current observation.

Output Requirements

Select and output one action from the provided Action List as the next optimal action to execute.
Format your response strictly as: character (e.g., A)
"""
    if with_example:
        system_prompt += "\nExamples\n"
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
    print(get_system_prompt(use_vlm=True, selection=True, with_example=False)[0])