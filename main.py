from environment import IsaacSim, read_yaml
import os
import sys

def test():
    root = 'experiment_log'
    env_type = "hard_env"
    env_num = 1
    f = open(os.path.join(root, f"{env_type}_{env_num}_base_1.txt"), 'w')
    sys.stdout = f
    config = read_yaml("config.yaml", env_type=env_type, env_num=env_num)
    Environment = IsaacSim(env_cfg_dict=config)
    Environment.test_pipeline()
    f.close()
    # Environment.data_collection()
    

if __name__ == "__main__":
    test()
