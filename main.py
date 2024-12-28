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
    
def run(mode, config, task_type, env_idx, root, affordance_type=None, score_metric=None, threshold=None):
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
        Environment = IsaacSim(env_cfg_dict=config, log_folder=log_folder, record_video=(mode == 'pipeline'), affordance_type=affordance_type)
        if mode == 'pipeline':
            Environment.test_pipeline(config.get('answer', []), score_metric=score_metric, threshold=threshold)
        elif mode == 'llm':
            Environment.test_llm()
            pyautogui.screenshot().save(os.path.join(log_folder, "result.jpg"))

    else:
        Environment = IsaacSim(env_cfg_dict=config)
        Environment.test_pipeline(action_sequence_answer=config['answer'])
                     
def experiments(mode, config_file, root, specific_task=[], affordance_type=None, score_metric=None, threshold=None):
    task_types = get_task_type_list(config_file)
    for task_type in task_types:
        for env_idx in range(1, get_task_env_num(config_file, task_type)+1):
            if not specific_task or (task_type, env_idx) in specific_task:
                config = read_yaml(config_file, task_type=task_type, env_idx=env_idx)
                run(mode, config, task_type, env_idx, root, affordance_type, score_metric=score_metric, threshold=threshold)

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
    config_file = os.environ.get('CONFIG_FILE', 'src/config/pdm.yaml')
    experiments('pipeline', config_file, root, affordance_type='lap', score_metric=1, threshold=0.1)
