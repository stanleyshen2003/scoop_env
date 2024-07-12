import cv2
from affordance.agent import Affordance_agent
from semantic import get_semantic_openai

class Decision_pipeline():
    def __init__(self, init_object_list) -> None:
        self.affordance_agent = Affordance_agent(init_object_list)
    def get_combine(
            self,
            instruction: str, 
            observation_rgb: str, 
            observation_d: str, 
            object_list=None, 
            tool_list=None, 
            action_sequence=None, 
            use_affordance=True
        ):
        action_list = ["scoop", "fork", "cut", "stir", "put_food", "DONE"]
        action_list.extend([f"take_tool ({tool})" for tool in tool_list])
        action_list.extend([f"put_tool ({tool})" for tool in tool_list])
        action_list.extend([f"move_to_{object.split(' (')[0]}" for object in object_list])
        rgb_img = cv2.imread(observation_rgb)
        gray_scale_img = cv2.imread(observation_d, cv2.IMREAD_GRAYSCALE)
        
        # get affordance score
        affordance = self.affordance_agent.get_affordance(rgb_img, gray_scale_img, action_list, action_sequence) if use_affordance else {action: 1 for action in action_list}
            
        # get semantic score
        semantic = get_semantic_openai(instruction, object_list, action_list, action_sequence)
        
        
        print(instruction)
        use_affordance and print(f"affordance {max(affordance, key=affordance.get)}")
        use_affordance and print(affordance)
        print(f"semantic {max(semantic, key=semantic.get)}")
        print(semantic)
        combined = {action: affordance[action] * semantic[action] for action in action_list}
        print(f"combined\n{combined}")
        # move_destination = None
        # if combined["move"] == max(combined, key=combined.get):
        #     instruction += "move to "
        #     move_destination = get_semantic_destination(instruction, object_list)
        
        return combined
    
if __name__ == "__main__":
    observation_rgb = 'affordance/data/spoon/30/0_rgb/000.png'
    observation_d = observation_rgb.replace('_rgb', '_depth')
    instruction = "Stir the beans in the bowl, then scoop it to the round plate. 1."
    decision_pipeline = Decision_pipeline()
    combined_score = decision_pipeline.get_combine(instruction, observation_rgb, observation_d)
    print(f"combined_score {max(combined_score, key=combined_score.get)}")
    print(combined_score)