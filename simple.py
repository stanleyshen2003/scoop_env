from simple_env import IsaacSim, read_yaml
from UserDefinedSettings import UserDefinedSettings
import argparse

import numpy as np



def test():
    config = read_yaml("config.yaml", env_num=2)
    weighingEnvironment = IsaacSim(env_cfg_dict=config)
    weighingEnvironment.data_collection()



if __name__ == "__main__":
    test()
