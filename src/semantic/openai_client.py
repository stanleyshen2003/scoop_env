import os
import socket
from openai import OpenAI
import pickle
import torch
import numpy as np

from typing import List

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
    # Input: {executed_action}
    return ret

def get_action_description_prompt():
    action_descrption = {
        'take_tool (tool name)': 'take the tool from the tool holder, there must be empty of the robotics hand', 
        'put_tool (tool_name)': 'put the tool back to the tool holder.', 
        'move_to_container': 'move to the container for further action like pulling or scooping', 
        'scoop': 'scoop the food, the speed of scooping will be affected by food state.', 
        'stir': 'stir the food.', 
        'put_food': 'put the food on your tool into the container.', 
        'pull_bowl_closer': 'pull the nearest bowl to the center of the table', 
        'DONE': 'indicates that the instruction is done.'
    }
    
    action_description_prompt = '\n'.join([f'{i + 1}. {action_name}: {action_descrption[action_name]}' for i, action_name in enumerate(action_descrption)])
    return action_description_prompt

def get_system_prompt(use_vlm=False, selection=False, scenario_description=False):
    if scenario_description:
        scenario_description_prompt = "Describe the food manipulation table top scenario from the image. Including what the robot are holding, spoon, knife, fork, or None"
        return scenario_description_prompt
    
    example_instruction = "Use knife to cut the food and fork it into the empty bowl, then put some beans on the food."
    example_action_seq = [
        "take_tool (knife)", 
        "move_to_white_cutting_board", 
        "cut", "put_tool (knife)", 
        "take_tool (fork)", 
        "move_to_white_cutting_board", 
        "fork", 
        "move_to_blue_bowl", 
        "put_food", 
        "put_tool (fork)", 
        "take_tool (spoon)", 
        "move_to_yellow_bowl", 
        "scoop",
        "move_to_blue_bowl", 
        "put_food", 
        "put_tool (spoon)", 
        "DONE"
    ]
    example_container_list = [
        "blue_bowl (empty)", 
        "white_cutting_board (with butter)", 
        "yellow_bowl (with green beans)", 
        "white_round_plate (empty)"
    ]
    example_action_list = [
        "put_tool (spoon)", 
        "put_tool (fork)", 
        "put_tool (knife)", 
        "take_tool (knife)", 
        "take_tool (fork)", 
        "take_tool (spoon)", 
        "move_to_blue_bowl", 
        "move_to_yellow_bowl", 
        "move_to_white_cutting_board", 
        "move_to_white_round_plate", 
        "cut", 
        "fork", 
        "scoop", 
        "put_food", 
        "pull bowl closer", 
        "DONE"
    ]
    example_action_seq = [action.replace('(', '').replace(')', '').replace('_', ' ') for action in example_action_seq]
    example_action_list = [action.replace('(', '').replace(')', '').replace('_', ' ') for action in example_action_list]
    example_container_list = [container.replace('_', ' ') for container in example_container_list]
    example_action_dict = format_action_choices(example_action_list)
    if selection:
        example_action_seq = [example_action_dict[action] for action in example_action_seq]
        example_action_list = [f'{selection}. {action}' for action, selection in example_action_dict.items()]
    
    vlm_prompt = """
You should consider the information from the input image and decide the appropriate action. For example, if there are only a few beans in the bowl, making it unsuitable for scooping, avoid scooping from that bowl.
You should avoid sccoping beans from the bowl too far away, since it may cause to failed.
If the primitive might cause to collision of failure, you may pull bowl to avoid it.
"""
    system_prompt = f"""You are a robot arm in food manipulation scneario. You should focus on your gripper. You need to pick an action from the action list to finish the whole task step by step.
Please also take the previous actions into consideration when choosing the next action.
{use_vlm * vlm_prompt}
{get_action_description_prompt()}

Example:
    Action list: {example_action_list}
    Initial object list: {example_container_list}
    Instruction: {example_instruction}
    {generate_prompt(example_action_seq, indent=True)}"""
    
    return system_prompt
        
def get_user_prompt(
    instruction,
    action_seq,
    action_dict,
    container_list,
) -> str:
    
    container_list = [container.replace('_', ' ') for container in container_list]
    action_seq = [action.replace('(', '').replace(')', '').replace('_', ' ') for action in action_seq]
    
    action_choices = [f"{v}. {k}" for k, v in action_dict.items()]
    action_seq_choices = [action_dict[action] for action in action_seq]
    user_prompt = f"""
Action list: {action_choices}
Initial object list: {container_list}
Instruction: {instruction}
Please choose one action from the action list to execute at the next iteration and output it directly.
{generate_prompt(action_seq_choices)}Iteration {len(action_seq_choices)+1}:
    Output: """
    return user_prompt
    
def format_action_choices(action_list: List[str]):
    return {action: chr(ord('A') + i) for i, action in enumerate(action_list)}

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
 
    openai_client = OpenAI()  
    
    action_description = {action.replace('(', '').replace(')', '').replace('_', ' '): action for action in action_list}
    action_dict = format_action_choices(list(action_description.keys()))
    system_prompt = get_system_prompt(use_vlm)
    user_prompt = get_user_prompt(instruction, action_seq, action_dict, container_list)
    system_content = [{"type": "text", "text": system_prompt}]
    user_content = [{"type": "text", "text": user_prompt}]
    if use_vlm:
        assert obs_url is not None, "Observation url could not be None"
        scenario_description = openai_client.chat.completions.create(
            # model='gpt-3.5-turbo',
            model='gpt-4o',
            messages=[
                {"role": "system", "content": [{"type": "text", "text": get_system_prompt(scenario_description=True)}]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": obs_url, "detail": "high"}}]}  
            ],
            logprobs=False
        ).choices[0].message.content
        user_content.append({"type": "image_url", "image_url": {"url": obs_url, "detail": "high"}})
    else:
        scenario_description = ''
        
    messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
    ]
    
    response_content = openai_client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4o',
        messages=messages,
        logprobs=True
    ).choices[0].message.content
    
    explanation_prompt = messages
    explanation_prompt[1]["content"][0]["text"] += f"{response_content} \nPlease explain why you choose the action."
    explanation = openai_client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4o',
        messages=explanation_prompt,
        logprobs=True
    ).choices[0].message.content
    
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
    
    system_content = [{"type": "text", "text": system_prompt}]
    user_content = [{"type": "text", "text": user_prompt}]
    if use_vlm:
        assert obs_image is not None, "Observation url could not be None"
        scenario_description = openai_client.chat.completions.create(
            # model='gpt-3.5-turbo',
            model='gpt-4o',
            messages=[
                {"role": "system", "content": [{"type": "text", "text": get_system_prompt(scenario_description=True)}]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": obs_image, "detail": "high"}}]}  
            ],
            logprobs=False
        ).choices[0].message.content
        user_content.append({"type": "image_url", "image_url": {"url": obs_image, "detail": "high"}})
    else:
        scenario_description = ''
        
    messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
    ]
    
    response = openai_client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4o',
        messages=messages,
        logprobs=True,
        top_logprobs=20
    )
    response_content = response.choices[0].message.content
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    top_logprobs = {top_logprob.token: top_logprob.logprob for top_logprob in top_logprobs}
    
    explanation_prompt = messages
    explanation_prompt[1]["content"][0]["text"] += f"{response_content} \nPlease explain why you choose the action."
    explanation = openai_client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4o',
        messages=explanation_prompt
    ).choices[0].message.content
    
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