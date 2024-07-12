from simple_env import IsaacSim, read_yaml
from UserDefinedSettings import UserDefinedSettings
import argparse
from PIL import Image
import os
import sys

def test():
    root = 'experiment_log'
    env_type = "hard_env"
    env_num = 1
    f = open(os.path.join(root, f"{env_type}_{env_num}_base_1.txt"), 'w')
    sys.stdout = f
    config = read_yaml("config.yaml", env_type=env_type, env_num=env_num)
    weighingEnvironment = IsaacSim(env_cfg_dict=config)
    weighingEnvironment.test_pipeline()
    f.close()
    # weighingEnvironment.data_collection()
    

if __name__ == "__main__":
    test()
