import os
import subprocess
from tqdm import tqdm

root = './experiment_log/gt_new_task'
video_list = [os.path.join(dirname, filename) for dirname, _, filenames in os.walk(root) for filename in filenames if filename.endswith('.mp4')]
for input_file in tqdm(video_list):
    output_file = input_file.replace('.mp4', '_temp.mp4')
    command = [
        "ffmpeg",
        "-i", input_file,  # Input file
        "-c:v", "libx264",  # Video codec
        "-c:a", "aac",  # Audio codec
        output_file  # Output file
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        continue
    os.replace(output_file, input_file)