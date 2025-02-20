from environment import IsaacSim
import os
import dotenv
import pyautogui
import cv2
import numpy as np
import time
import datetime
import logging

from src.config import *
dotenv.load_dotenv()

def get_log_id(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')

def data_collection(config):
    Environment = IsaacSim(env_cfg_dict=config)
    Environment.data_collection()
    
def run(mode, config, task_type, env_idx, root, test_type=None, threshold=None):
    assert mode in ['llm', 'pipeline', 'calibration_collection'], f"Invalid mode {mode}"
    print("=" * 10, task_type.upper(), env_idx, "=" * 10)
    
    if mode in ['llm', 'pipeline']:
        log_dir = os.path.join(root, f"{task_type}_{env_idx}")
        log_id: str = str(get_log_id(log_dir))
        log_folder = os.path.join(log_dir)
        log_folder = os.path.join(log_dir, log_id)
        for folder in os.listdir(log_dir):
            if os.path.exists(os.path.join(log_dir, folder, 'result_sequence.txt')):
                return
        os.makedirs(log_folder, exist_ok=True)
        Environment = IsaacSim(env_cfg_dict=config, log_folder=log_folder, record_video=(mode == 'pipeline'))
        if mode == 'pipeline':
            Environment.test_pipeline(config.get('answer', []), test_type=test_type, threshold=threshold, use_vlm=True)
        elif mode == 'llm':
            Environment.test_llm()
            pyautogui.screenshot().save(os.path.join(log_folder, "result.jpg"))

    else:
        Environment = IsaacSim(env_cfg_dict=config)
        Environment.test_pipeline(action_sequence_answer=config['answer'])
                     
def experiments(mode, config_file, root, specific_task=[], test_type=None, threshold=None):
    task_types = get_task_type_list(config_file)
    for task_type in task_types:
        for env_idx in range(1, get_task_env_num(config_file, task_type)+1):
            if not specific_task or (task_type, env_idx) in specific_task or task_type in specific_task:
                config = read_yaml(config_file, task_type=task_type, env_idx=env_idx)
                print(f"Running {task_type} {env_idx}")
                run(mode, config, task_type, env_idx, root, test_type=test_type, threshold=threshold)

def calibration():
    config_root = 'src/config'
    config_name = 'pdm_calibration'
    config_file = os.path.join(config_root, f"{config_name}.yaml")
    print(config_file)
    for task_type in get_task_type_list(config_file)[4:]:
        for env_idx in range(1, get_task_env_num(config_file, task_type)+1):
            config = read_yaml(config_file, task_type=task_type, env_idx=env_idx)
            run('calibration_collection', config, task_type, env_idx)
            

if __name__ == "__main__":
    root = os.environ.get('RESULT_DIR', 'experiment_log/test')
    config_file = os.environ.get('CONFIG_FILE', 'src/config/config.yaml')
    test_type = os.environ.get('TEST_TYPE', None)
    threshold = os.environ.get('THRESHOLD', None)
    task_type = os.environ.get('TASK_TYPE', None)
    env_idx = os.environ.get('ENV_IDX', 1)
    env_idx = int(env_idx) if env_idx else 1
    
    # specific_task = [('amount_ambiguity', 1), ('amount_ambiguity', 2), ('spatial_proximity', 1), ('spatial_proximity', 2)]
    all_task = get_task_type_list(config_file)
    excepted_task = ['mix_type', 'general_hard', 'spatial_relationship', 'distance']
    specific_task = list(set(all_task) - set(excepted_task))
    # experiments('pipeline', config_file, root, test_type=test_type, threshold=threshold, specific_task=specific_task)
    
    mode = 'pipeline'
    if not specific_task or (task_type, env_idx) in specific_task or task_type in specific_task:
        config = read_yaml(config_file, task_type=task_type, env_idx=env_idx)
        print(f"Running {task_type} {env_idx}")
        run(mode, config, task_type, env_idx, root, test_type=test_type, threshold=threshold)
