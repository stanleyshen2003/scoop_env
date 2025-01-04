import os
import cv2
from src.affordance.agents import *
from src.semantic import *
from src.utils import *

def get_action_list(tool_list, object_list):
    action_list = ["scoop", "stir", "put_food", "pull_bowl_closer", "DONE"]
    action_list.extend([f"take_tool ({tool})" for tool in tool_list])
    action_list.extend([f"put_tool ({tool})" for tool in tool_list])
    action_list.extend([f"move_to_{object.split(' (')[0]}" for object in object_list])
    return action_list

affordance_agent_list = {
    "classifier": Affordance_agent_classifier,
    "lap": Affordance_agent_LAP,
    "ours": Affordance_agent_ours
} 
class Decision_pipeline():
    def __init__(self, init_object_list, tool_list, log_folder) -> None:
        self.init_object_list = init_object_list
        self.action_list = get_action_list(tool_list, init_object_list)
        self.log_folder = log_folder
        self.obs_id = 0
        self.affordance_agent = Affordance_agent(self.init_object_list, self.action_list)
    
    def set_affordance_agent(self, affordance_type):
        self.affordance_agent = affordance_agent_list.get(affordance_type, Affordance_agent)(self.init_object_list, self.action_list)
    
    def get_score(
        self,
        instruction: str, 
        observation_rgb_path: str, 
        observation_d_path: str, 
        action_sequence=None,
        use_vlm=False,
        action_candiadate=[],
        affordance_only=False,
        semantic_only=False
    ):
        # action_list = ["scoop", "fork", "cut", "stir", "put_food", "pull_bowl_closer", "DONE"]
        # action_list = get_action_list(tool_list, object_list)
        # print(action_list)
        
        # get affordance score
        if semantic_only:
            affordance = {action: 1 for action in self.action_list}
        else:
            affordance = self.affordance_agent.get_affordance(observation_rgb_path, observation_d_path, action_sequence, action_candiadate)
            affordance = sort_scores_dict(affordance)
            open(os.path.join(self.log_folder, f"affordance_{self.obs_id}.txt"), 'w').write(f"{affordance}")
            
        # get semantic score
        if affordance_only:
            semantic = {action: 1 for action in self.action_list}
        else:
            obs_image = None
            base64_image = encode_image(observation_rgb_path)
            if use_vlm:
                obs_image = base64_image
                cv2.imwrite(os.path.join(self.log_folder, f'observation_{self.obs_id}.png'), cv2.imread(observation_rgb_path))
            semantic = get_selection_score_openai(instruction, self.init_object_list, self.action_list, action_sequence, use_vlm, obs_image, log_folder=self.log_folder, obs_id=self.obs_id)
        # semantic = get_semantic_gemini(instruction, object_list, action_list, action_sequence)
        
        
        score = {action: affordance[action] * semantic[action] for action in self.action_list}
        score = sort_scores_dict(score)
        open(os.path.join(self.log_folder, f"combined_{self.obs_id}.txt"), 'w').write(f"{score}")
        
        self.obs_id += 1
        print(instruction)
        print("=" * 20)
        print(f"affordance {max(affordance, key=affordance.get)}")
        print(affordance)
        print("=" * 20)
        print(f"semantic {max(semantic, key=semantic.get)}")
        print(semantic)
        print("=" * 20)
        print(f"combined\n{score}")
        print("=" * 20)
        
        return score
    
if __name__ == "__main__":
    observation_rgb_path = 'affordance/data/spoon/30/0_rgb/000.png'
    observation_d_path = observation_rgb_path.replace('_rgb', '_depth')
    instruction = "Stir the beans in the bowl, then scoop it to the round plate. 1."
    decision_pipeline = Decision_pipeline()
    combined_score = decision_pipeline.get_score(instruction, observation_rgb_path, observation_d_path)
    print(f"combined_score {max(combined_score, key=combined_score.get)}")
    print(combined_score)