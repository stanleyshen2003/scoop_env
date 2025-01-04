import os
from ultralytics import YOLOWorld
def get_yolo_prob(rgb_img_path, class_names):
    model = YOLOWorld("/home/hcis-s17/multimodal_manipulation/scoop_env/src/affordance/ours/models/yolov8x-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes
    results = model.predict(rgb_img_path)
    image_root = '/home/hcis-s17/multimodal_manipulation/scoop_env/src/affordance/ours/image'
    image_id = len(os.listdir(image_root))
    results[0].save(filename=os.path.join(image_root, f'{image_id}.jpg'))
    
if __name__ == '__main__':
    rgb_img_path = '/home/hcis-s17/multimodal_manipulation/scoop_env/src/affordance/classifier/data/spoon/2/0_rgb/021.png'
    get_yolo_prob(rgb_img_path, ['spoon', 'fork', 'knife'])