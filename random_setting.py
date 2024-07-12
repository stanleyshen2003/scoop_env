from simple_env import IsaacSim
import matplotlib.colors as mcolors
import random
from datetime import datetime
import os
from PIL import Image
import threading
import yaml


class Collector():
    def __init__(self, root):
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)
            with open(os.path.join(self.root, 'store_keys.yaml'), 'w') as f:
                yaml.dump({'fork': 0, 'knife': 0, 'spoon': 0}, f)
        with open(os.path.join(self.root, 'store_keys.yaml'), 'r') as f:
            self.now_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.config = {}
        self.random_config()
        self.env = IsaacSim(self.config)
        

    def random_config(self):
        random.seed(datetime.now().timestamp())
        grains_colors = ['darkgreen', 'darkred', 'black']
        colors = [i for i in mcolors.CSS4_COLORS.keys() if i not in grains_colors]
        
        tools = ['fork', 'knife', 'spoon']
        config = {}
        config['franka_type'] = 'original'
        config['output_folder'] = 'test_random'
        assert config['output_folder'] is not None
        
        # select tool to hold on hand and tool to put in env
        selected_tool_list = []
        take_tool_idx = random.randint(0, 4)
        primary_tool = tools[take_tool_idx] if take_tool_idx < 2 else 'spoon'
        selected_tool_list.append(primary_tool)
        for tool in tools:
            if tool != primary_tool and random.randint(0,1):
                selected_tool_list.append(tool)
        random.shuffle(selected_tool_list)
        config['tool'] = selected_tool_list
        config['primary_tool'] = primary_tool
        
        
        # select number of bowls
        bowl_num = random.choice([1, 2, 3, 4])
        plate_num = random.choice([0, 1]) if bowl_num != 4 else 0
        cutting_board_num = random.choice([0, 1]) if bowl_num+plate_num != 4 else 0
        container_list = []
        
        print(f'bowl num: {bowl_num}')
        print(f'plate num: {plate_num}')
        print(f'cutting board num: {cutting_board_num}')
        
        
        for _ in range(cutting_board_num):
            container = {}
            container['x'] = 0.35 + (0.02) * random.random()
            container['y'] = -0.2 + 0.03 * random.random()
            container['type'] = 'round_plate'
            container['container_color'] = random.choice(colors)
            if config['primary_tool'] != 'spoon':
                container['food'] = random.choice(['cuttable_food', 'forked_food'])
            else:
                container['food'] = random.choice(['cuttable_food', 'forked_food', "None"])
            container['color'] = ['black']
            container['amount'] = 1600
            container_list.append(container)
            
        for i in range(bowl_num):
            container = {}
            container['x'] = "None"
            container['y'] = "None"
            container['type'] = 'bowl'
            container['container_color'] = random.choice(colors)
            if i == 0:
                container['food'] = 'ball'
            else:
                container['food'] = random.choice(['ball', "None"])
            container['color'] = [random.choice(grains_colors)]
            container['amount'] = random.randint(400, 1600)
            container_list.append(container)
        for _ in range(plate_num):
            container = {}
            container['x'] = "None"
            container['y'] = "None"
            container['type'] = 'round_plate'
            container['container_color'] = random.choice(colors)
            if config['primary_tool'] != 'spoon':
                container['food'] = random.choice(['cuttable_food', 'forked_food'])
            else:
                container['food'] = random.choice(['cuttable_food', 'forked_food', "None"])
            container['color'] = ['black']
            container['amount'] = 1600
            container_list.append(container)
            
        config['containers'] = container_list
        self.config = config
        

    def collect_one(self):
        for key in self.now_dict.keys():
            os.makedirs(os.path.join(self.root, key), exist_ok=True)
        
        rgb_images_0, depth_images_0, rgb_images_1, depth_images_1 = self.env.take_tool_move_around()
        
        
        os.makedirs(os.path.join(self.root, self.config["primary_tool"], str(self.now_dict[self.config["primary_tool"]])), exist_ok=True)
        os.makedirs(os.path.join(self.root, self.config["primary_tool"], str(self.now_dict[self.config["primary_tool"]]), '0_rgb'), exist_ok=True)
        os.makedirs(os.path.join(self.root, self.config["primary_tool"], str(self.now_dict[self.config["primary_tool"]]), '0_depth'), exist_ok=True)
        os.makedirs(os.path.join(self.root, self.config["primary_tool"], str(self.now_dict[self.config["primary_tool"]]), '1_rgb'), exist_ok=True)
        os.makedirs(os.path.join(self.root, self.config["primary_tool"], str(self.now_dict[self.config["primary_tool"]]), '1_depth'), exist_ok=True)
        
        
        for i in range(len(rgb_images_0)):
            Image.fromarray(rgb_images_0[i]).save(os.path.join(self.root, self.config["primary_tool"], str(self.now_dict[self.config["primary_tool"]]),'0_rgb', "{:03d}".format(i)+'.png'))
            Image.fromarray(depth_images_0[i]).save(os.path.join(self.root, self.config["primary_tool"], str(self.now_dict[self.config["primary_tool"]]), '0_depth', "{:03d}".format(i)+'.png'))
            
        for i in range(len(rgb_images_1)):
            Image.fromarray(rgb_images_1[i]).save(os.path.join(self.root, self.config["primary_tool"], str(self.now_dict[self.config["primary_tool"]]),'1_rgb', "{:03d}".format(i)+'.png'))
            Image.fromarray(depth_images_1[i]).save(os.path.join(self.root, self.config["primary_tool"], str(self.now_dict[self.config["primary_tool"]]), '1_depth', "{:03d}".format(i)+'.png'))
        
        self.now_dict[self.config["primary_tool"]] += 1
        
        with open(os.path.join(self.root, 'store_keys.yaml'), 'w') as f:
            yaml.dump(self.now_dict, f)

     
    




if __name__ == "__main__":
    root = 'hello'
    collector = Collector(root)
    collector.collect_one()
    
