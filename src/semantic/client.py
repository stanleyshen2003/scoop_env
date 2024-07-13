import socket
import pickle
from typing import List

def generate_prompt(action_seq, indent=False):
    ret = ""
    for i, action in enumerate(action_seq):
        executed_action = " ".join([f'{j+1}. {action_seq[j]}' for j in range(i)])
        if indent:
            ret += f"""Iteration {i+1}: {action}
    """
        else:
            ret += f"""Iteration {i+1}: {action}
"""
    # Input: {executed_action}
    return ret

def action_description_prompt():
    action_list = ['take tool tool_name', 'move to container', 'scoop', 'fork', 'cut', 'stir', 'put food', 'put tool tool_name', 'DONE']
    action_description_prompt = f"""Here is some explanation of the actions in the action list:
    1. {action_list[0]}: take the tool from the tool holder.
    2. {action_list[1]}: move to the container.
    3. {action_list[2]}: scoop the food.
    4. {action_list[3]}: fork the food.
    5. {action_list[4]}: cut the food.
    6. {action_list[5]}: stir the food.
    7. {action_list[6]}: put the food on your tool into the container.
    8. {action_list[7]}: put the tool back to the tool holder.
    9. {action_list[8]}: indicates that the instruction is done."""
    return action_description_prompt

def get_semantic(instruction: str, container_list=None, action_list=["scoop", "fork", "cut", "move", "stir", "DONE"], action_seq=None):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('140.113.215.115', 9999))
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
    data = {
        "dialogs": [
        [
            {"role": "system", "content": f"""You need to pick an action from the action list to finish the whole task step by step.
Please also take the previous actions into consideration when choosing the next action.
{action_description_prompt()}

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
You should also take the previous actions into consideration when choosing actions.
Please choose one action from the action list to execute at iteration {len(action_seq)+1}?
Please ONLY tell me the name of the action.
{generate_prompt(action_seq)}Iteration {len(action_seq)+1}: """
            }
        ]
    ],
        "temperature": 0.6,
        "max_gen_len": 50,
        "action_list": action_description,
        "object_list": container_list,
    }
    # print(data["dialogs"][0])

    # Serialize the data
    # print(data["dialogs"][0][0]["content"])
    data_bytes = pickle.dumps(data)

    # Send data to the server
    client.send(data_bytes)
    client.shutdown(socket.SHUT_WR)

    # Receive response from the server
    response = client.recv(4096 * 4)
    client.close()

    if response:
        # Deserialize the response
        response_data = pickle.loads(response)
        # print(response_data)
        
        _semantic = response_data[0]
        semantic = {}
        for key, value in _semantic.items():
            semantic[action_description_dict[key]] = value
        return semantic
    return None


if __name__ == '__main__':
    instruction = "Stir the beans in the bowl, then scoop it to the round plate."
    object_list = ["red_bowl (empty)", "white_round_plate (empty)", "green_bowl (with beans)"]
    action_list = ['take_tool (spoon)', 'take_tool (fork)', 'move_to_green_bowl', 'move_to_red_bowl', 'move_to_white_round_plate', 'scoop', 'fork', 'cut', 'move', 'stir', 'put_food', 'DONE']
    semantic = get_semantic(instruction, object_list, action_list=action_list, action_seq=['take_tool (spoon)'])
    print(semantic)
