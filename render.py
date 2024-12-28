import os
from environment import IsaacSim
from src.config import *
def render_all_config(config_file):
    task_types = get_task_type_list(config_file)
    file_root = config_file.replace('.yaml', '')
    os.makedirs(file_root, exist_ok=True)
    for task_type in task_types:
        for env_idx in range(1, get_task_env_num(config_file, task_type)+1):
            config = read_yaml(config_file, task_type=task_type, env_idx=env_idx)
            img_filename = os.path.join(file_root, f"{task_type}_{env_idx}.jpg")
            txt_filename = os.path.join(file_root, f"{task_type}_{env_idx}.txt")
            if not os.path.exists(img_filename):
                Environment = IsaacSim(env_cfg_dict=config)
                answer_sequence = '\n'.join(config.get('answer', []))   
                Environment.render_config(img_filename, text=f"{config.get('instruction', '')}\n{answer_sequence}")
if __name__ == '__main__':
    config_file = 'src/config/pdm.yaml'
    render_all_config(config_file)