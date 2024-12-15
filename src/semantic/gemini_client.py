import socket
from openai import OpenAI
import pickle
import torch
import numpy as np
import os
import dotenv
import google.ai.generativelanguage_v1 as genai

from .utils import *
genai.GenerationConfig

def get_semantic(instruction: str, container_list=None, action_list=["scoop", "fork", "cut", "move", "stir", "DONE"], action_seq=None):
    dotenv.load_dotenv()

    gemini_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=gemini_key)
    gemini_client = genai.GenerativeModel('gemini-1.5-pro')    
    example_instruction = "Use knife to cut the food and fork it into the empty bowl, then put some beans on the food."
    example_action_seq = ["take_tool (knife)", "move_to_white_cutting_board", "cut", "put_tool (knife)", "take_tool (fork)", "move_to_white_cutting_board", "fork", "move_to_blue_bowl", "put_food", "put_tool (fork)", "take_tool (spoon)", "move_to_yellow_bowl", "scoop","move_to_blue_bowl", "put_food", "put_tool (spoon)", "DONE"]
    example_container_list = ["blue_bowl (empty)", "white_cutting_board (with butter)", "yellow_bowl (with green beans)", "white_round_plate (empty)"]
    example_action_list = ["put_tool (spoon)", "put_tool (fork)", "put_tool (knife)", "take_tool (knife)", "take_tool (fork)", "take_tool (spoon)", "move_to_blue_bowl", "move_to_yellow_bowl", "move_to_white_cutting_board", "move_to_white_round_plate", "cut", "fork", "scoop", "put_food", "DONE"]
    example_action_seq = [action.replace('(', '').replace(')', '').replace('_', ' ') for action in example_action_seq]
    example_action_list = [action.replace('(', '').replace(')', '').replace('_', ' ') for action in example_action_list]
    example_container_list = [container.replace('_', ' ') for container in example_container_list]
    container_list = [container.replace('_', ' ') for container in container_list]
    action_seq = [action.replace('(', '').replace(')', '').replace('_', ' ') for action in action_seq]
    
    # for container in example_container_list:
    #     example_action_seq.append(f"move_to_{container}")
    #     example_action_seq.append(f"move_to_{container.split(' (')[0]}")
    
    action_description_dict = {action.replace('(', '').replace(')', '').replace('_', ' '): action for action in action_list}
    action_description = list(action_description_dict.keys())
    messages = [
            {"role": "system", "content": f"""You need to pick an action from the action list to finish the whole task step by step.
Please also take the previous actions into consideration when choosing the next action.
{get_action_description_prompt()}

Example:
    Action list: {example_action_list}
    Initial object list: {example_container_list}
    Instruction: {example_instruction}
    {generate_prompt(example_action_seq, indent=True)}"""
            },
            {"role": "user", "content": f"""
Action list: {action_description}
Initial object list: {container_list}
Instruction: {instruction}
Please choose one action from the action list to execute at the next iteration and output it directly.
{generate_prompt(action_seq)}Iteration {len(action_seq)+1}:
    Output:"""
            }
    ]
    
    response = gemini_client.generate_content(messages).text
    
    print(response)
    
    _semantic = {}
    for action in action_description:
        # socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # socket_client.connect(('140.113.215.115', 9999))
        # data = {"action": action, "choice": response}
        # data_bytes = pickle.dumps(data)

        # # Send data to the server
        # socket_client.send(data_bytes)
        # socket_client.shutdown(socket.SHUT_WR)
        # _bert_score = socket_client.recv(4096 * 4)
        # bert_score = pickle.loads(_bert_score) if _bert_score else None
        # _semantic[action] = np.log(bert_score)
        same = [x if x in response.split() else None for x in action.split()]
        while None in same:
            same.remove(None)
        _semantic[action] = len(same) / len(response.split())
        # socket_client.close()
    scores = torch.softmax(torch.Tensor(list(_semantic.values())), dim=-1).cpu().tolist()
    # scores = list(_semantic.values())
    # _scores = np.array(list(_semantic.values()))
    # scores = (_scores - _scores.mean()) / np.std(_scores)
    # scores = torch.softmax(torch.from_numpy(scores), dim=-1).cpu().tolist()
    
    _semantic = {action: score for action, score in zip(_semantic.keys(), scores)}
    # _semantic = {action: score([action], [response], lang='en')[-1].item() for action in action_description}
    # _semantic = {action: 0.7 if action == response else 0.1 for action in action_description}
    semantic = {}
    for key, val in _semantic.items():
        semantic[action_description_dict[key]] = val
    # print(semantic)
    
    return semantic

if __name__ == '__main__':
    instruction = "Stir the beans in the bowl, then scoop it to the round plate."
    object_list = ["red_bowl (empty)", "white_round_plate (empty)", "green_bowl (with beans)"]
    action_list = ['take_tool (spoon)', 'take_tool (fork)', 'move_to_green_bowl', 'move_to_red_bowl', 'move_to_white_round_plate', 'scoop', 'fork', 'cut', 'move', 'stir', 'put_food', 'DONE']
    semantic = get_semantic(instruction, object_list, action_list=action_list, action_seq=['take_tool (spoon)', 'move_to_green_bowl', 'stir', 'scoop', 'move_to_white_round_plate', 'put_food', 'put_tool (spoon)'])
    print(semantic)