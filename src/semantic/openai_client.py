import os
import socket
from openai import OpenAI
import pickle
import torch
import numpy as np

from typing import List
from .utils import *

def call_openai_api(messages, model='gpt-4o'):
    openai_client = OpenAI()  
    content = openai_client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model=model,
        messages=messages,
        logprobs=True,
        top_logprobs=20
    )
    return content

def get_semantic(
    instruction: str, 
    container_list=None, 
    action_list=["scoop", "fork", "cut", "move", "stir", "DONE"], 
    action_seq=None, 
    use_vlm=False, 
    obs_url=None,
    log_folder=None,
    obs_id=None,
    )->dict:
 
    
    action_description = {action.replace('(', '').replace(')', '').replace('_', ' '): action for action in action_list}
    action_dict = format_action_choices(list(action_description.keys()))
    system_prompt = get_system_prompt(use_vlm)
    user_prompt = get_user_prompt(instruction, action_seq, action_dict, container_list)
    system_content = [{"type": "text", "text": system_prompt}]
    user_content = [{"type": "text", "text": user_prompt}]
    if use_vlm:
        assert obs_url is not None, "Observation url could not be None"
        scenario_prompt = get_messages(get_system_prompt(scenario_description=True), image_url=obs_url)
        scenario_description = call_openai_api(scenario_prompt).choices[0].message.content
        messages = get_messages(system_prompt, user_prompt, obs_url)
    else:
        scenario_description = ''
        messages = get_messages(system_prompt, user_prompt)
    
    response_content = call_openai_api(messages).choices[0].message.content
    
    explanation_prompt = get_messages(system_prompt, user_prompt + f"{response_content} \nPlease explain why you choose the action.", obs_url)
    explanation = call_openai_api(explanation_prompt).choices[0].message.content
    
    print(response_content)
    if use_vlm:
        print(scenario_description)
    print(explanation)
    
    _semantic = {}
    for action in action_description:
        # socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # socket_client.connect(('140.113.215.115', 9999))
        # data = {"action": action, "choice": response_content}
        # data_bytes = pickle.dumps(data)

        # # Send data to the server
        # socket_client.send(data_bytes)
        # socket_client.shutdown(socket.SHUT_WR)
        # _bert_score = socket_client.recv(4096 * 4)
        # bert_score = pickle.loads(_bert_score) if _bert_score else None
        # _semantic[action] = np.log(bert_score)
        same = [x if x in response_content.split() else None for x in action.split()]
        while None in same:
            same.remove(None)
        _semantic[action] = len(same) / len(response_content.split())
        # socket_client.close()
    scores = torch.softmax(torch.Tensor(list(_semantic.values())), dim=-1).cpu().tolist()
    # scores = list(_semantic.values())
    # _scores = np.array(list(_semantic.values()))
    # scores = (_scores - _scores.mean()) / np.std(_scores)
    # scores = torch.softmax(torch.from_numpy(scores), dim=-1).cpu().tolist()
    
    _semantic = {action: score for action, score in zip(_semantic.keys(), scores)}
    # _semantic = {action: score([action], [response_content], lang='en')[-1].item() for action in action_description}
    # _semantic = {action: 0.7 if action == response_content else 0.1 for action in action_description}
    semantic = {}
    for key, val in _semantic.items():
        semantic[action_description[key]] = val
    # print(semantic)
    
    if log_folder is not None:
        assert obs_id is not None, "Please provide observation id"
        with open(os.path.join(log_folder, f'prompt_{obs_id}.txt'), 'w') as f:
            f.write('\n'.join(['\n[SYS]\n' + system_prompt, '\n[USER]\n' + user_prompt, '\n[DES]\n' + scenario_description, '\n[RES]\n' + response_content, '\n[EXP]\n' + explanation]))
    return sorted(semantic, key=lambda x: x.value(), reverse=True)

def get_selection_score(
    instruction: str, 
    container_list=None, 
    action_list=["scoop", "move", "stir", "DONE"],
    action_seq=None, 
    use_vlm=False, 
    obs_image=None,
    log_folder=None,
    obs_id=None,
    )->dict:
    print("instruction", instruction)
    print("container_list", container_list)
    print("action_list", action_list)
    print("action_seq", action_seq)
    print("use_vlm", use_vlm)
    print("log_folder", log_folder)
    print("obs_id", obs_id)
 
    openai_client = OpenAI() 
    
    action_description = {action.replace('(', '').replace(')', '').replace('_', ' '): action for action in action_list}
    action_dict = format_action_choices(list(action_description.keys()))
    system_prompt = get_system_prompt(use_vlm)
    user_prompt = get_user_prompt(instruction, action_seq, action_dict, container_list)
    
    if use_vlm:
        assert obs_image is not None, "Observation url could not be None"
        scenario_prompt = get_messages(get_system_prompt(scenario_description=True), '', obs_image)
        scenario_description = call_openai_api(scenario_prompt).choices[0].message.content
        messages = get_messages(system_prompt, user_prompt, obs_image)
    else:
        scenario_description = ''
        messages = get_messages(system_prompt, user_prompt)
    
    response = call_openai_api(messages)
    response_content = response.choices[0].message.content
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    top_logprobs = {top_logprob.token: top_logprob.logprob for top_logprob in top_logprobs}
    
    explanation_prompt = get_messages(system_prompt, user_prompt + f"{response_content} \nPlease explain why you choose the action.", obs_image)
    explanation = call_openai_api(explanation_prompt).choices[0].message.content
    
    print(messages[0]["content"][0]['text'])
    print('=' * 80)
    print(messages[1]["content"][0]['text'])
    print(top_logprobs)
    print(response_content)
    if use_vlm:
        print(scenario_description)
    print(explanation)
    
    _semantic = {action: top_logprobs.get(choice, float('-inf')) for action, choice in action_dict.items()}
    semantic = {}
    for key, val in _semantic.items():
        semantic[action_description[key]] = val
    semantic = dict(sorted(semantic.items(), key=lambda x: x[1], reverse=1))
    # print(semantic)
    
    if log_folder is not None:
        assert obs_id is not None, "Please provide observation id"
        with open(os.path.join(log_folder, f'prompt_{obs_id}.txt'), 'w') as f:
            f.write('\n'.join([
                '\n[SYS]\n' + system_prompt, 
                '\n[USER]\n' + user_prompt, 
                '\n[DES]\n' + scenario_description, 
                '\n[RES]\n' + response_content, 
                '\n[EXP]\n' + explanation, 
                '\n[SCORE]\n' + f"{semantic}"
            ]))
    return semantic

def get_calibration_data(
    instruction: str, 
    action: str,
    action_list: List[str],
    container_list=None, 
    action_seq=None,
    use_vlm=False
) -> str:
    """
    Given action and instruction to get question and answer pair in text for confidence calibration
    """
    container_list = [container.replace('_', ' ') for container in container_list]
    action_seq = [action.replace('(', '').replace(')', '').replace('_', ' ') for action in action_seq]
    
    action_description = {action.replace('(', '').replace(')', '').replace('_', ' '): action for action in action_list}
    action_dict = format_action_choices(list(action_description.keys()))
    
    system_prompt = get_system_prompt(use_vlm, True)
    user_prompt = get_user_prompt(instruction, action_seq, action_dict, container_list)
    answer = action_dict[action.replace('(', '').replace(')', '').replace('_', ' ')]
    return system_prompt, user_prompt, answer
    
    
if __name__ == '__main__':
    instruction = "Stir the beans in the bowl, then scoop it to the round plate."
    object_list = ["red_bowl (empty)", "white_round_plate (empty)", "green_bowl (with beans)"]
    action_list = ['take_tool (spoon)', 'take_tool (fork)', 'put_tool (spoon)', 'put_tool (fork)', 'move_to_green_bowl', 'move_to_red_bowl', 'move_to_white_round_plate', 'scoop', 'fork', 'cut', 'move', 'stir', 'put_food', 'DONE']
    action_seq = ['take_tool (spoon)', 'move_to_green_bowl', 'stir', 'scoop', 'move_to_white_round_plate', 'put_food', 'put_tool (spoon)']
    print("SELECTION\n\n")
    system_prompt, user_prompt, answer = get_calibration_data(instruction, 'take_tool (fork)', action_list, object_list, action_seq, True)
    print(system_prompt)
    print(user_prompt)
    print(answer)