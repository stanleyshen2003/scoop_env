import math
import yaml
import matplotlib.colors as mcolors
import numpy as np
import random
import sys
import inspect

base_color = ['white', 'black', 'red', 'green', 'purple', 'pink', 'orange', 'brown', 'blue', 'yellow', 'gray']
hsv_list = {c: mcolors.rgb_to_hsv(mcolors.to_rgb(c)) for c in mcolors.CSS4_COLORS.keys()}
color_list = {}
for c in list(mcolors.CSS4_COLORS):
    # c contains the base color name
    for base_color_name in base_color:
        if base_color_name in c:
            color_list[c] = base_color_name
            break
    else:
        closet_color = min(base_color, key=lambda x: np.linalg.norm(np.array(hsv_list[x]) - np.array(hsv_list[c])))
        color_list[c] = closet_color
bean_color_list = {
    'darkred': 'red',
    'darkgreen': 'mung',
    'black': 'chocolate',
    'peru': 'speckled'
}
color_list = {
    'violet': 'purple', 
    'cornflowerblue': 'blue', 
    'pink': 'pink', 
    'white': 'white', 
    'slategrey': 'grey', 
    'darkred': 'red', 
    'papayawhip': 'white', 
    'mediumpurple': 'purple', 
    'antiquewhite': 'white', 
    'darkslateblue': 'blue', 
    'darkgoldenrod': 'yellow', 
    'navy': 'blue', 
    'lightgrey': 'white', 
    'coral': 'orange', 
    'deepskyblue': 'blue', 
    'palegreen': 'green', 
    'indigo': 'purple', 
    'seashell': 'white'
}
DISTANCE_RANGE = (0.35, 0.6), (-0.2, 0.25)
DISTANCE_RANGE_SCOOPABLE = (0.35, 0.55), (-0.15, 0.25)
DISTANCE_RANGE_BLOCK_HOLDER = (0.35, 0.40), (-0.26, -0.24)
DISTANCE_RANGE_BLOCK_DUMBWAITER = (0.38, 0.5), (0.32, 0.34)
DISTANCE_RANGE_PULL = (0.71, 0.72), (-0.15, 0.2)
DISTANCE_RANGE_TOO_FAR = (0.72, 0.8), (-0.2, 0.25)
DISTANCE_RANGE_TOO_CLOSE = (0.25, 0.4), (0.2, 0.3)
DISTANCE_RANGE_PULL_CENTER = (0.4, 0.5), (-0.1, 0.1)

DISTANCE_RANGE_PULLABLE_SET = set([DISTANCE_RANGE_BLOCK_HOLDER, DISTANCE_RANGE_BLOCK_DUMBWAITER, DISTANCE_RANGE_PULL])
DISTANCE_RANGE_HEATABLE_SET = set([DISTANCE_RANGE_BLOCK_DUMBWAITER])


SCOOP_PUT_SEQUENCE = 'move_to_{scoop_target1}_bowl', 'scoop', 'move_to_{put_target1}_bowl', 'drop_food'
PULL_SEQUENCE = 'move_to_{pull_target}_bowl', 'pull_bowl_closer'
HEAT_SEQUENCE = 'open_dumbwaiter', 'move_to_{heat_target}_bowl', 'put_bowl_into_dumbwaiter', 'close_dumbwaiter', 'start_dumbwaiter'

task_list = {
    ## container
    ## [(x1, x2), (y1, y2), food_amount, scoop/put]
    'general_easy': {
        'containers': [
            [*DISTANCE_RANGE, 1600, True],
            [*DISTANCE_RANGE, 0, True],
        ],
        'random_containers': [
            [*DISTANCE_RANGE, 1500, True],
            [*DISTANCE_RANGE, 0, True],
            None, # used to sample nothing
        ],
        'answer': [
            ('', ['grasp_spoon', *SCOOP_PUT_SEQUENCE, 'put_spoon_back', 'DONE']),
            ('', ['grasp_spoon', *SCOOP_PUT_SEQUENCE, *SCOOP_PUT_SEQUENCE, 'put_spoon_back', 'DONE']),
            # ('', ['grasp_spoon', *SCOOP_PUT_SEQUENCE, 'move_to_{scoop_target2}_bowl', 'scoop', 'move_to_{put_target2}_bowl', 'drop_food', 'put_spoon_back', 'DONE']),
        ]
    }, 
    'general_hard': {
    
    }, 
    'amount_ambiguity': {
        'containers': [
            # [(x1, x2), (y1, y2), food_amount]
            [*DISTANCE_RANGE_SCOOPABLE, 1600, True],
            [*DISTANCE_RANGE_SCOOPABLE, 0, True],
        ],
        'random_containers': [
            [*DISTANCE_RANGE, 200, False],
            [*DISTANCE_RANGE, 400, False],
            [*DISTANCE_RANGE, 800, False],
        ],
        'answer': [
            ('', ['grasp_spoon', *SCOOP_PUT_SEQUENCE, 'put_spoon_back', 'DONE']),
        ]
    }, 
    'spatial_proximity': {
        'containers': [
            [*DISTANCE_RANGE_SCOOPABLE, 1600, True],
            [*DISTANCE_RANGE_SCOOPABLE, 0, True],
        ],
        'random_containers': [
            [*DISTANCE_RANGE_TOO_CLOSE, 1600, False],
            [*DISTANCE_RANGE_TOO_FAR, 1600, False],
            [*DISTANCE_RANGE_TOO_FAR, 0, False],
        ],
        'answer': [
            ('', ['grasp_spoon', *SCOOP_PUT_SEQUENCE, 'put_spoon_back', 'DONE']),
        ]
    }, 
    'distance': {
        'containers': [
            [*DISTANCE_RANGE_PULL, 1600, True],
            [*DISTANCE_RANGE, 0, True],
        ],
        'answer': [
            ('', [*PULL_SEQUENCE, 'grasp_spoon', *SCOOP_PUT_SEQUENCE, 'put_spoon_back', 'DONE']),
        ]
    }, 
    'obstacles_holder_1': {
        'containers': [
            [*DISTANCE_RANGE_BLOCK_HOLDER, 1600, True],
            [(0.35, 0.55), (0.15, 0.25), 0, True],
        ],
        'random_containers': [
            [*DISTANCE_RANGE_SCOOPABLE, 1600, True],
        ],
        'answer': [
            ('', [*PULL_SEQUENCE, 'grasp_spoon', *SCOOP_PUT_SEQUENCE, 'put_spoon_back', 'DONE']),
        ]
        
    },
    'obstacles_holder_2': {
        'containers': [
            [*DISTANCE_RANGE_BLOCK_HOLDER, 0, True],
            [(0.35, 0.55), (0.15, 0.25), 1600, True],
        ],
        'random_containers': [
            [*DISTANCE_RANGE_SCOOPABLE, 0, True],
        ],
        'answer': [
            ('', [*PULL_SEQUENCE, 'grasp_spoon', *SCOOP_PUT_SEQUENCE, 'put_spoon_back', 'DONE']),
        ]
    },
    'obstacles_dumbwaiter': {
        'containers': [
            [(0.35, 0.55), (-0.15, 0.), 1600, True],
            [*DISTANCE_RANGE_BLOCK_DUMBWAITER, 0, True],
        ],
        'answer': [
            ('', ['grasp_spoon', *SCOOP_PUT_SEQUENCE, 'put_spoon_back', *PULL_SEQUENCE, *HEAT_SEQUENCE, 'DONE']),
        ]
    },
}
        
def get_random_container_color():
    color_code = random.choice(list(color_list))
    color_name = color_list[color_code]
    return color_name, color_code

def get_random_bean_color():
    color_code = random.choice(list(bean_color_list))
    color_name = bean_color_list[color_code]
    return color_name, color_code

def get_random_position(x_range, y_range):
    return np.random.uniform(*x_range), np.random.uniform(*y_range)

def valid_container_position(position1, position2, task_type, min_container_dist=0.2, pull_container_center=(0.45, 0)):
    def calculate_dist(pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    if task_type == 'distance' and abs(position1[1] - position2[1]) < min_container_dist:
        return False
    # if 'obstacles' in task_type and abs(position1[0] - position2[0]) < min_container_dist:
    #     return False
    if calculate_dist(position1, position2) < min_container_dist:
        return False
    return True


def generate_config(task_type, num=10):
    config_path = f'src/config/{task_type}.yaml'
    _config_list = {}
    idx = 1 
    sub_task_types = [key for key, val in task_list.items() if task_type in key and len(val) > 0]
    print(sub_task_types)
    assert len(sub_task_types) > 0, f"{task_type} doesn't exist"
    
    for i, sub_task_type in enumerate(sub_task_types):
        sub_task_list = task_list[sub_task_type]
        while True:
            print(idx)
            if idx > (i + 1) * num // len(sub_task_types):
                break
            container_configs = []
            container_configs.extend(sub_task_list['containers'])
            if len(sub_task_list.get('random_containers', [])) > 0:
                random_container_num = random.randint(1, min(len(sub_task_list['random_containers']), 3))
                random_container_configs = random.sample(sub_task_list['random_containers'], random_container_num)
                if None not in random_container_configs:
                    container_configs.extend(random_container_configs)
                    
            instruction = ""
            tool = ['spoon']
            containers = []
            scoop_container = []
            put_container = []
            pull_container = []
            heat_container = []
            
            for container_config in container_configs:
                stop_sample = False
                attempt = 0
                max_attempt = 100
                distance_range = tuple(container_config[:2])
                while not stop_sample:
                    pos = get_random_position(*distance_range)
                    stop_sample = True
                    for _container in containers:
                        if not valid_container_position(pos, [_container['x'], _container['y']], sub_task_type):
                            stop_sample = False
                            attempt += 1
                            break
                    if attempt > max_attempt:
                        break
                if not stop_sample:
                    break
                container = {}
                container['x'], container['y'] = pos
                container['type'] = 'bowl'
                stop_sample = False
                while not stop_sample:
                    color, colorcode = get_random_container_color()
                    stop_sample = True
                    for _container in containers:
                        if _container['color'] == color:
                            stop_sample = False
                            break
                container['color'] = color
                container['colorcode'] = colorcode
                
                if distance_range in DISTANCE_RANGE_PULLABLE_SET:
                    pull_container.append(color)
                if distance_range in DISTANCE_RANGE_HEATABLE_SET:
                    heat_container.append(color)
                    
                if container_config[2] != 0:
                    bean_color, bean_colorcode = get_random_bean_color()
                    container['food'] = {
                        'type': 'ball',
                        'color': [bean_color],
                        'colorcode': [bean_colorcode],
                        'amount': container_config[2],
                        'position': [1, 1]
                    }
                    if container_config[3]:
                        scoop_container.append(color)
                else:
                    container['food'] = {
                        'type': 'None',
                        'color': [],
                        'colorcode': [],
                        'amount': 0,
                        'position': [1, 1]
                    }
                    if container_config[3]:
                        put_container.append(color)
                containers.append(container)
            if len(containers) != len(container_configs):
                continue
            answer = random.choice(sub_task_list.get('answer', [('', ['DONE'])]))
            instruction, answer = answer
            target_container = {'scoop_target1': random.choice(scoop_container), 'put_target1': random.choice(put_container)}
            if len(pull_container) > 0:
                target_container['pull_target'] = random.choice(pull_container)
            if len(heat_container) > 0:
                target_container['heat_target'] = random.choice(heat_container)
            answer = ' '.join(answer).format(**target_container).split()
            _config_list[idx] = {
                'instruction': instruction,
                'answer': answer,
                'tool': tool,
                'containers': containers
            }
            idx += 1
    config_list = {task_type: _config_list}
    with open(config_path, 'w') as f:
        yaml.dump(config_list, f, sort_keys=False, default_flow_style=None)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if 'all' in sys.argv:
            task_types = [task_type for task_type, config in task_list.items() if len(config) > 0]
        else:
            task_types = sys.argv[1:]
        for task_type in task_types:
            print(task_type)
            generate_config(task_type)