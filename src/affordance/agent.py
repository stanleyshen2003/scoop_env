import torch
import cv2
import numpy as np
from .model import ResNet50
from .preprocess import preprocess_single_image_cv2
import sys
from typing import List

class Affordance_agent():
    def __init__(self, init_object_list: List[str], model_path='src/affordance/model/49_0.9759231705667524.pth'):
        self.env_state = {obj.split(" (")[0]: obj.split(" (")[1].replace(")", "") != "empty" for obj in init_object_list}
        self.tool_on_hand = False
        self.food_on_hand = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet50().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def predict(self, rgb_img, gray_scale_img):
        rgb_img = preprocess_single_image_cv2(rgb_img, 1080, 1080, 256, 256)
        depth_img = preprocess_single_image_cv2(gray_scale_img, 1080, 1080, 256, 256)
        # cv2.imwrite('temp.png', rgb_img)
        # cv2.imwrite('temp_depth.png', depth_img)
        rgb_img = np.array(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), dtype=np.float32).transpose(2, 0, 1) / 255.0
        depth_img = np.expand_dims(np.array(depth_img, dtype=np.float32) / 255.0, axis=0)
        
        input_img = torch.cat((torch.tensor(rgb_img), torch.tensor(depth_img)), dim=0)
        input_img = input_img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_img).squeeze(0)

        logits = torch.softmax(logits[1:-1], dim=0)
        
        # dealing with None affordance
        # if logits[-1] > 0.7:
        #     logits = logits[:-1]
        #     # logits = torch.full_like(logits, 0.1)
        # else:
        #     logits = logits[1:-1]
        #     logits = (logits + 1) / 2
        return logits.cpu().numpy()
    
    def get_affordance(
            self, 
            rgb_img, 
            gray_scale_img, 
            action_list: List[str], 
            action_seq: List[str]
        ):
        last_action = action_seq[-1] if len(action_seq) else None
        if last_action:
            if not self.tool_on_hand and "take_tool" in last_action:
                self.tool_on_hand = True
            elif self.tool_on_hand and "put_tool" in last_action:
                self.tool_on_hand = False
            elif not self.food_on_hand and "scoop" in last_action and "put" not in last_action:
                self.food_on_hand = True
            elif self.food_on_hand and "put" in last_action:
                self.food_on_hand = False
                if len(action_seq) >= 2 and "move" in action_seq[-2]:
                    self.env_state[last_action.replace("move_to_", "")] = True

        action_list_affordance = ["scoop", "fork", "cut", "stir", "DONE"]
        logits = self.predict(rgb_img, gray_scale_img)
        affordance_scores = logits.tolist()
        affordance_scores.append(affordance_scores[1]) # stir
        affordance_scores.append(0.5) # DONE
        _affordance = {action: score for action, score in zip(action_list_affordance, affordance_scores)}
        
        tool_score_high = 0.5
        tool_score_low = 0.1
        move_score_high = 0.5
        move_score_low = 0.1
        put_food_score_high = 0.5
        put_food_score_low = 0.1
        
        put_tool_score = tool_score_low + int(self.tool_on_hand) * (tool_score_high - tool_score_low)
        take_tool_score = tool_score_high - put_tool_score
        move_score = move_score_high
        put_food_score = put_food_score_low + int(self.food_on_hand) * (put_food_score_high - put_food_score_low)
        
        affordance = {}
        for action in action_list:
            if action in _affordance.keys(): 
                affordance[action] = _affordance[action]
            elif "take_tool" in action:
                affordance[action] = take_tool_score
            elif "put_tool" in action:
                affordance[action] = put_tool_score
            elif "put_food" in action:
                affordance[action] = put_food_score
            elif "move" in action:
                affordance[action] = move_score - int(not self.food_on_hand and not self.env_state[action.replace("move_to_", "")]) * (move_score_high - move_score_low)
            
        return affordance
        
        
            
if __name__ == '__main__':
    agent = Affordance_agent()
    rgb_img = cv2.imread('data/spoon/0/0_rgb/076.png')
    gray_scale_img = cv2.imread('data/spoon/0/0_depth/076.png', cv2.IMREAD_GRAYSCALE)
    logits = agent.predict(rgb_img, gray_scale_img)
    print(logits)