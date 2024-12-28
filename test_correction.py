
import os
import socket
from openai import OpenAI
import pickle
import torch
import numpy as np
import datetime

from typing import List
    
from src.utils import *
        
def call_openai_api(messages, model='gpt-4o'):
    openai_client = OpenAI()
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        logprobs=False,
        temperature=0.1
    )
    response_content = response.choices[0].message.content
    return response_content


def generate_prompt(action_seq, indent=False):
    ret = ""
    for i, action in enumerate(action_seq):
        executed_action = " ".join([f'{j+1}. {action_seq[j]}' for j in range(i)])
        if indent:
            ret += f"""Iteration {i+1}:
        Output: {action}
    """
        else:
            ret += f"""Iteration {i+1}:
    Output: {action}
"""
    return ret


def format_action_choices(action_list: List[str]):
    return {action: chr(ord('A') + i) for i, action in enumerate(action_list)}

class Prompt():
    def __init__(self):
        self.generate_action_description_prompt()
    
    def generate_action_description_prompt(self):
        action_list = ['take_tool (tool name)', 'move_to_{color}_{container}', 'scoop', 'fork', 'cut', 'stir', 'put_food', 'put_tool (tool_name)', 'pull_bowl_closer', 'DONE']
        self.action_description_prompt = f"""Here is some explanation of the actions in the action list:
1. {action_list[0]}: take the tool from the tool holder, there must be empty of the robotics hand
2. {action_list[1]}: move to the container specified by color and container.
3. {action_list[2]}: scoop the food, the execution time will be strongly affected by the state of food.
4. {action_list[3]}: fork the food.
5. {action_list[4]}: cut the food.
6. {action_list[5]}: stir the food.
7. {action_list[6]}: put the food on your tool into the container.
8. {action_list[7]}: put the tool back to the tool holder.
9. {action_list[8]}: pull the nearest bowl to the center of the table
10. {action_list[9]}: indicates that the instruction is done."""
        
    def next_action_prompt(self, action_seq, instruction, object_list=None):

        system_prompt = f"""You are a smart assistant tasked with identifying what food properties or information should be take into consideration before deciding the next action for a robot about a food manipulation task. Consider all previous actions and their outcomes when deciding. Please do not decide the action and specify what you should achieve in the next step and what you should take into consideration to achieve the goal in high level. Provide your answer in 50 words or fewer.
{self.action_description_prompt}"""
        action_sequence = ", ".join([f"{i+1}. {action}" for i, action in enumerate(action_seq)])
        user_prompt = f"""Instruction: {instruction}
Previous actions:
{action_sequence}"""
        return system_prompt, user_prompt
    
    def extract_important_information_prompt(self, instruction, important_considerations, object_list=None):
        system_prompt = f"""You are a great observer that can describe the environment in detail. Given an image of the food manipulation scenario, the overall goal, the object list, and the key considerations of determining the next move, extract the important information from the image and the instruction. Please provide your answer in 50 words or fewer."""
        user_prompt = f"""Overall goal: {instruction}
Key considerations: {important_considerations}
Object list: {', '.join(object_list)}"""
        
        return system_prompt, user_prompt
    
    def extract_from_choice_prompt(self, instruction, action_sequence, choices, container_list=None):
        system_prompt = f"""You are a smart assistant tasked with identifying what food properties (extrinsic, amount, distribution) or information that should be take into consideration in a food manipulation task. You will be given some candidate actions, please think through all the choices as thorough as possible and list what I should know in order to make the choice. Please do not choose the action directly.
{self.action_description_prompt}"""
        action_sequence = ", ".join([f"{i+1}. {action}" for i, action in enumerate(action_seq)])
        user_prompt = f"""Instruction: {instruction}
Previous actions: {action_sequence}
Object list: {', '.join(container_list)}
Possible next actions: {', '.join(choices)}"""
        return system_prompt, user_prompt
        
    

def get_messages(system_prompt, user_prompt, image_url=None):
    system_content = [{"type": "text", "text": system_prompt}]
    if image_url is not None:
        user_content = [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}]
    else:
        user_content = [{"type": "text", "text": user_prompt}]
    messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
    ]
    return messages

'''
This is the first version
1. summarize the important information to make the next decision given the instruction and the current action sequence (no image)
2. extract the important information from the image to make the next decision given the instruction and the current action sequence (with image)
3. make decision! (not done in this function)
'''
def get_selection_score1(
    instruction: str, 
    container_list=None, 
    action_list=["scoop", "fork", "cut", "move", "stir", "DONE"], 
    action_seq=None, 
    use_vlm=False, 
    obs_url=None,
    log_folder=None,
    obs_id=None,
    )->dict:
 
    prompts = Prompt()
    
    container_list = [container.replace('_', ' ') for container in container_list]
    action_seq = [action.replace('(', '').replace(')', '').replace('_', ' ') for action in action_seq]
    
    
    ## Get the next action description
    system_prompt, user_prompt = prompts.next_action_prompt(action_seq, instruction)   
    messages = get_messages(system_prompt, user_prompt)
    next_action_description = call_openai_api(messages)
    
    print(next_action_description)
    
    ## Logging
    os.makedirs(log_folder, exist_ok=True)
    with open(os.path.join(log_folder, "1_next_action_description.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + next_action_description
        ]))
        
    ## Get the important information from the image
    system_prompt, user_prompt = prompts.extract_important_information_prompt(instruction, next_action_description)
    messages = get_messages(system_prompt, user_prompt, obs_url)
    important_information = call_openai_api(messages)
    
    print(important_information)
    
    ## Logging
    with open(os.path.join(log_folder, "2_important_information.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + important_information
        ]))
    
    return
    
'''
This is the second version

'''

def get_selection_score2(
    instruction: str, 
    container_list=None, 
    action_list=["scoop", "fork", "cut", "move", "stir", "DONE"], 
    action_seq=None, 
    use_vlm=False, 
    obs_url=None,
    log_folder=None,
    obs_id=None,
    possible_options=None,
    )->dict:
 
    prompts = Prompt()
    
    container_list = [container.replace('_', ' ') for container in container_list]
    action_seq = [action.replace('(', '').replace(')', '').replace('_', ' ') for action in action_seq]
    
    
    ## Get the next action description
    system_prompt, user_prompt = prompts.extract_from_choice_prompt(instruction, action_seq, possible_options, container_list)   
    messages = get_messages(system_prompt, user_prompt)
    next_action_description = call_openai_api(messages)
    
    print(next_action_description)
    
    ## Logging
    os.makedirs(log_folder, exist_ok=True)
    with open(os.path.join(log_folder, "1_next_action_description.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + next_action_description
        ]))
    
    ## Get the important information from the image
    system_prompt, user_prompt = prompts.extract_important_information_prompt(instruction, next_action_description, container_list)
    messages = get_messages(system_prompt, user_prompt, obs_url)
    important_information = call_openai_api(messages)
    
    print(important_information)
    
    ## Logging
    with open(os.path.join(log_folder, "2_important_information.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + important_information
        ]))
    
    return
 

if __name__ == '__main__':
    instruction = "I need one scoop of green beans, and I'm in a hurry. Can you make it quick?"
    container_list = ['white_bowl (with green beans)', 'red_bowl (with green beans)', 'skyblue_bowl (with green beans)', 'blue_bowl (with black beans)', 'yellow_bowl (empty)']
    action_list = ['scoop', 'fork', 'cut', 'stir', 'put_food', 'pull_bowl_closer', 'DONE', 'take_tool (spoon)', 'take_tool (fork)', 'take_tool (knife)', 'put_tool (spoon)', 'put_tool (fork)', 'put_tool (knife)', 'move_to_white_bowl', 'move_to_red_bowl', 'move_to_skyblue_bowl', 'move_to_blue_bowl', 'move_to_yellow_bowl']
    action_seq = ['take_tool (spoon)']
    use_vlm = True
    log_folder = "experiment_log/correction2/" + '{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now())
    obs_id = 1
    obs_url = encode_image('experiment_log/hard_9/20241120000205/observation_1.png')
    possible_options = ['move_to_white_bowl', 'move_to_red_bowl', 'move_to_skyblue_bowl']
    get_selection_score2(instruction, container_list, action_list, action_seq, use_vlm, obs_url=obs_url, log_folder=log_folder, obs_id=obs_id, possible_options=possible_options)