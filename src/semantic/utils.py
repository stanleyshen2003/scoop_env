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
        "cut", 
        "put_tool (knife)", 
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
You should avoid sccoping beans from the bowl too far away, or pull it closer before you scoop it, since it may cause to failed.
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
        
def get_user_prompt(instruction, action_seq, action_dict, container_list) -> str:
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

def next_action_prompt(instruction, action_seq):
    system_prompt = f"""You are a smart assistant tasked with identifying what food properties or information should be take into consideration before deciding the next action for a robot about a food manipulation task. Consider all previous actions and their outcomes when deciding. Please do not decide the action and specify what you should achieve in the next step and what you should take into consideration to achieve the goal in high level. Provide your answer in 50 words or fewer.
{get_action_description_prompt()}"""
    action_sequence = ", ".join([f"{i+1}. {action}" for i, action in enumerate(action_seq)])
    user_prompt = f"""Instruction: {instruction}
Previous actions:
{action_sequence}"""
    return system_prompt, user_prompt

def extract_important_information_prompt(instruction, important_considerations, object_list=None):
    system_prompt = f"""You are a great observer that can describe the environment in detail. Given an image of the food manipulation scenario, the overall goal, the object list, and the key considerations of determining the next move, extract the important information from the image and the instruction. Please provide your answer in 50 words or fewer."""
    user_prompt = f"""Overall goal: {instruction}
Key considerations: {important_considerations}
Object list: {', '.join(object_list)}"""
    return system_prompt, user_prompt

def extract_from_choice_prompt(instruction, action_seq, choices, container_list=None):
    system_prompt = f"""You are a smart assistant tasked with identifying what food properties (extrinsic, amount, distribution) or information that should be take into consideration in a food manipulation task. You will be given some candidate actions, please think through all the choices as thorough as possible and list what I should know in order to make the choice. Please do not choose the action directly.
{get_action_description_prompt()}"""
    action_sequence = ", ".join([f"{i+1}. {action}" for i, action in enumerate(action_seq)])
    user_prompt = f"""Instruction: {instruction}
Previous actions: {action_sequence}
Object list: {', '.join(container_list)}
Possible next actions: {', '.join(choices)}"""
    return system_prompt, user_prompt
    
def format_action_choices(action_list: List[str]):
    return {action: chr(ord('A') + i) for i, action in enumerate(action_list)}


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