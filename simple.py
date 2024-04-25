from simple_env import IsaacSim, read_yaml
from UserDefinedSettings import UserDefinedSettings
import argparse
from PIL import Image
import os

def test():
    
    config = read_yaml("config.yaml", env_type="hard_env", env_num=1)
    weighingEnvironment = IsaacSim(env_cfg_dict=config)
    rgb_images, depth_images = weighingEnvironment.data_collection()
    

if __name__ == "__main__":
    test()
