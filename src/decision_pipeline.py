import os
import cv2
from src.affordance.agent import Affordance_agent
from src.semantic import *
import base64
import mimetypes


def encode_image(image_path: str):
    """Encodes an image to base64 and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"

def get_action_list(tool_list, object_list):
    action_list = ["scoop", "stir", "put_food", "pull_bowl_closer", "DONE"]
    action_list.extend([f"take_tool ({tool})" for tool in tool_list])
    action_list.extend([f"put_tool ({tool})" for tool in tool_list])
    action_list.extend([f"move_to_{object.split(' (')[0]}" for object in object_list])
    return action_list
    
class Decision_pipeline():
    def __init__(self, init_object_list, log_folder=None) -> None:
        self.affordance_agent = Affordance_agent(init_object_list)
        self.log_folder = log_folder
        self.obs_id = 0
    def get_combine(
            self,
            instruction: str, 
            observation_rgb: str, 
            observation_d: str, 
            object_list=None, 
            tool_list=None, 
            action_sequence=None, 
            use_affordance=True,
            use_vlm=False,
        ):
        # action_list = ["scoop", "fork", "cut", "stir", "put_food", "pull_bowl_closer", "DONE"]
        action_list = get_action_list(tool_list, object_list)
        print(action_list)
        rgb_img = cv2.imread(observation_rgb)
        gray_scale_img = cv2.imread(observation_d, cv2.IMREAD_GRAYSCALE)
        
        # get affordance score
        affordance = self.affordance_agent.get_affordance(rgb_img, gray_scale_img, action_list, action_sequence) if use_affordance else {action: 1 for action in action_list}
            
        # get semantic score
        obs_image = None
        base64_image = encode_image(observation_rgb)
        if use_vlm:
            obs_image = base64_image
            if self.log_folder is not None:
                cv2.imwrite(os.path.join(self.log_folder, f'observation_{self.obs_id}.png'), rgb_img)
        semantic = get_selection_score_openai(instruction, object_list, action_list, action_sequence, use_vlm, obs_image, log_folder=self.log_folder, obs_id=self.obs_id)
        # semantic = get_semantic_gemini(instruction, object_list, action_list, action_sequence)
        
        
        print(instruction)
        print("=" * 20)
        use_affordance and print(f"affordance {max(affordance, key=affordance.get)}")
        use_affordance and print(affordance)
        print("=" * 20)
        print(f"semantic {max(semantic, key=semantic.get)}")
        print(semantic)
        print("=" * 20)
        combined = {action: affordance[action] * semantic[action] for action in action_list}
        print(f"combined\n{combined}")
        print("=" * 20)
        # move_destination = None
        # if combined["move"] == max(combined, key=combined.get):
        #     instruction += "move to "
        #     move_destination = get_semantic_destination(instruction, object_list)
        
        self.obs_id += 1
        
        return combined
    
if __name__ == "__main__":
    observation_rgb = 'affordance/data/spoon/30/0_rgb/000.png'
    observation_d = observation_rgb.replace('_rgb', '_depth')
    instruction = "Stir the beans in the bowl, then scoop it to the round plate. 1."
    decision_pipeline = Decision_pipeline()
    combined_score = decision_pipeline.get_combine(instruction, observation_rgb, observation_d)
    print(f"combined_score {max(combined_score, key=combined_score.get)}")
    print(combined_score)