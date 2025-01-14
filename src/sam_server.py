import torch
import cv2
import numpy as np
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import socket
import json

def segment_image(img_path, saved_path):
    checkpoint = "/home/hcis-s17/multimodal_manipulation/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    prompt_cord = np.array([
        (50, 1000), 
        (24, 500), 
        (102, 1500)
    ])
    prompt_label = np.array([1, 0, 0])
    print(img_path)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        img = cv2.imread(img_path)
        predictor.set_image(img)
        # masks, _, _ = predictor.predict(prompt_cord, prompt_label)
        masks, _, _ = predictor.predict()
    new_masks = np.zeros_like(img)
    colormap = [
        [0, 0, 255],  # red
        [0, 255, 0],  # green
        [255, 0, 0],  # blue
        [255, 255, 0],  # cyan
        [255, 0, 255],  # magenta
        [0, 255, 255],  # yellow
    ]
    print(len(masks), ' masks')
    for i, mask in enumerate(masks):
        new_masks[mask > 0] = np.array(colormap[i], dtype=np.uint8)

    # add transparent mask
    img = 0.8 * new_masks + img
    cv2.imwrite(saved_path, img)

def socket_handler(data):
    img_path = data['img_path']
    segment_image(img_path)
    
def main():
    host = '127.0.0.1'
    port = 12345
    
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)

        s.bind((host, port))
        s.listen(1)
        print(f"Listening on {host}:{port}")
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                data = conn.recv(4096)
                if not data:
                    break
                data_dict = json.loads(data.decode('utf-8'))
                img_path = data_dict['img_path']
                saved_path = f"{img_path.split('.')[0] + '_segment'}.jpg"
                try:
                    segment_image(img_path, saved_path)
                    conn.sendall(saved_path.encode('utf-8'))
                except FileNotFoundError:
                    conn.sendall('File not found'.encode('utf-8'))
                
if __name__ == '__main__':
    main()
