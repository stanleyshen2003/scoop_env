import os
from src.config import *
if __name__ == '__main__':
    root = os.environ.get('RESULT_DIR', 'experiment_result/test')
    config_file = os.environ.get('CONFIG_FILE', 'src/config/pdm.yaml')
    for task_type in get_task_type_list(config_file):
        for env_idx in range(1, get_task_env_num(config_file, task_type)+1):
            os.makedirs(os.path.join(root, f"{task_type}_{str(env_idx)}"), exist_ok=True)