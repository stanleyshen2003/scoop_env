from environment import IsaacSim, read_yaml
import os
import sys
import dotenv
import glob
import pyautogui
import cv2
import numpy as np
import threading
import time
import trace
import datetime
import logging

dotenv.load_dotenv()

Environment = None

class thread_with_trace(threading.Thread):
  def __init__(self, *args, **keywords):
    threading.Thread.__init__(self, *args, **keywords)
    self.killed = False

  def start(self):
    self.__run_backup = self.run
    self.run = self.__run      
    threading.Thread.start(self)

  def __run(self):
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup

  def globaltrace(self, frame, event, arg):
    if event == 'call':
      return self.localtrace
    else:
      return None

  def localtrace(self, frame, event, arg):
    if self.killed:
      if event == 'line':
        raise SystemExit()
    return self.localtrace

  def kill(self):
    self.killed = True

def run_experiment(config):
    global Environment
    Environment.test_pipeline()
    
def get_log_id(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    
def test(mode, env_type, env_num):
    assert mode in ['llm', 'pipeline', 'calibration_collection'], f"Invalid mode {mode}"
    print("=" * 10, env_type.upper(), env_num, "=" * 10)
    root = 'experiment_log'
    
    log_folder = None
    if mode != 'calibration_collection':
        log_dir = os.path.join(root, f"{env_type}_{env_num}")
        log_id: str = str(get_log_id(log_dir))
        log_folder = os.path.join(log_dir, log_id)
        os.makedirs(log_folder, exist_ok=True)
        # f = open(os.path.join(log_dir, f"{log_id}.txt"), 'w')
        # sys.stdout = f
        config = read_yaml("config.yaml", env_type=env_type, env_num=env_num)
        Environment = IsaacSim(env_cfg_dict=config, log_folder=log_folder)
        if mode == 'pipeline':
            Environment.test_pipeline()
        elif mode == 'llm':
            Environment.test_llm()
            pyautogui.screenshot().save(os.path.join(log_dir, f"{log_id}.jpg"))
    else:
        config = read_yaml("confidence_calibration/config.yaml", env_type=env_type, env_num=env_num)
        Environment = IsaacSim(env_cfg_dict=config)
        Environment.test_pipeline(action_sequence=config['answer'])
        
    # f.close()
    sys.stdout = sys.__stdout__
    # Environment.data_collection()

def data_collection(env_type, env_num):
    config = read_yaml("config.yaml", env_type=env_type, env_num=env_num)
    Environment = IsaacSim(env_cfg_dict=config)
    Environment.data_collection()

def experiments():
    root = 'experiment_log'
    env_types = ['simple_env', 'hard_env']
    env_nums = [i + 1 for i in range(5)]
    resolution = (1920, 1080)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 60.0
    max_time = 300 # sec
    global Environment
    for env_type in env_types:
        for env_num in env_nums:
            print(env_type, env_num, "\n" + "=" * 30)
            log_dir = os.path.join(root, f"{env_type}_{env_num}")
            log_id: str = str(get_log_id(log_dir))
            video_filename = os.path.join(log_dir, f"{log_id}.mp4")
            f = open(os.path.join(log_dir, f"{log_id}.txt"), 'w')
            # sys.stdout = f
            config = read_yaml("config.yaml", env_type=env_type, env_num=env_num)
            Environment = IsaacSim(config)
            t = thread_with_trace(target=run_experiment, kwargs={"config": config})
            out = cv2.VideoWriter(video_filename, codec, fps, resolution)
            t.start()
            start = time.time()
            while True:
                img = pyautogui.screenshot()
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
                if not t.is_alive() or time.time() - start > max_time:
                    t.kill()
                    print(time.time() - start)
                    print(t.is_alive())
                    Environment.gym.destroy_sim(Environment.sim)
                    Environment.gym.destroy_viewer(Environment.viewer)
                    break
            out.release()
            f.close()          

if __name__ == "__main__":
  # data_collection('simple', 2)
  # test('pipeline', 'medium', 6)
  test('calibration_collection', 'simple', 1)
  # test('llm', 'hard', 5)
  # env_types = ['hard', 'simple', 'medium']
  # env_nums = [i + 1 for i in range(5)]
  #for env_type in env_types:
    #for env_num in env_nums:
     # test('llm', env_type, env_num)
    # experiments()
    
