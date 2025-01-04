import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Patch


category = {
    'scoop': 'royalblue',
    'take tool': 'lightblue',
    'put tool': 'deepskyblue',
    'move to': 'purple',
    'pull bowl': 'navy',
    'other': 'red'
}
category_names = list(category.keys())
def main(txt_file, splitter='\t'):
    name = txt_file.split('.')[0]
    answer_list = open(txt_file, 'r').readlines()
    answer_list = [l.strip().split(splitter) for l in answer_list]
    empty_data = [[0 for __ in range(len(category_names))] for _ in range(22)]
    for answer in answer_list:
        if len(answer) != 4:
            continue
        for i, category_name in enumerate(category_names):
            if category_name in answer[1]:
                break
        else:
            i = -1
        empty_data[int(answer[3])][i] += 1
    empty_data = np.array(empty_data)
    bar_width = 0.35
    x = len(empty_data)
    fig, ax = plt.subplots(figsize=(8, 6))

# Plot stacked bars
    for rank, datas in enumerate(empty_data):
        bottom = 0
        for i, data in enumerate(datas):
            if data == 0:
                continue
            ax.bar(rank, data, bottom=bottom, label=category_names[i], color=category[category_names[i]])
            bottom += data


    # Add labels, title, and legend
    ax.set_xlabel('Ranks')
    ax.set_ylabel('Counts')
    ax.set_title(f'Ranks distribution of the ground truth action {name.upper()}')
    legend_patches = [Patch(color=v, label=k) for k, v in category.items()]
    ax.legend(handles=legend_patches)

    plt.tight_layout()
    plt.savefig(f"visualization_{name}.png")
    
    
if __name__ == '__main__':
    txt_files = ['answer_sys_2.txt'] if len(sys.argv) < 2 else sys.argv[1:]
    print(txt_files)
    for txt_file in txt_files:
        main(txt_file)