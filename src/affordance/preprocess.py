from glob import glob
import random
from PIL import Image
import cv2
import os
from tqdm import tqdm

def process_single_image(img, crop_w, crop_h, resize_w, resize_h):
    img_w, img_h = img.size
    img = img.crop((img_w//2 - crop_w//2, img_h//2 - crop_h//2, crop_w//2 + img_w//2, crop_h//2 + img_h//2))
    img = img.resize((resize_w, resize_h))
    return img

def preprocess_single_image_cv2(img, crop_w, crop_h, resize_w, resize_h):
    if len(img.shape) == 2:
        img_h, img_w = img.shape
    else:
        img_h, img_w, img_d = img.shape
    img = img[img_h//2 - crop_h//2:crop_h//2 + img_h//2, img_w//2 - crop_w//2:crop_w//2 + img_w//2]
    # img = img.resize((resize_w, resize_h))
    img = cv2.resize(img, (resize_w, resize_h))
    return img




if __name__ == '__main__':
    new_root = 'preprocessed_data'
    
    for tool in ["spoon", "fork", "knife"]:
        files = glob(f'data/{tool}/*/*_rgb/*')
        crop_w, crop_h = 1080, 1080
        resize_h, resize_w = 256,256


        for (i, file) in tqdm(enumerate(files), total=len(files)):
            # print(i)
            img = Image.open(file)
            # crop from the center of width and height
            process_single_image(img, crop_w, crop_h, resize_w, resize_h)
            # create a new directory
            os.makedirs(file.replace('data', new_root).replace(file.split('/')[-1], ''), exist_ok=True)
            img.save(file.replace('data', new_root))
            
            d_file = file.replace('_rgb', '_depth')
            img = Image.open(d_file)
            process_single_image(img, crop_w, crop_h, resize_w, resize_h)
            os.makedirs(d_file.replace('data', new_root).replace(d_file.split('/')[-1], ''), exist_ok=True)
            img.save(d_file.replace('data', new_root))