import os
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import sys
sys.path.append('..')

from src.utils import *
from src.semantic.utils import get_system_prompt, get_example_prompt
from utils import parse_list
openai_client = OpenAI()

def get_log_prob(system_content, user_content, answer, model, action_list, rank_in_action=False):
      
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    response = openai_client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model=model,
        messages=messages,
        logprobs=True,
        top_logprobs=20
    )
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    top_logprobs = {top_logprob.token: top_logprob.logprob for top_logprob in top_logprobs}
    sort_action_in_list = sorted(action_list, key=lambda x: top_logprobs.get(x, -1), reverse=True)
    sort_action = sorted(top_logprobs.items(), key=lambda x: x[1], reverse=True)
    sort_action = {k[0]: i + 1 for i, k in enumerate(sort_action)}
    sort_action_in_list = {k: i + 1 for i, k in enumerate(sort_action_in_list)}
    rank = sort_action.get(answer, -1)
    rank_in_action = sort_action_in_list.get(answer, -1)
    
    return top_logprobs.get(answer, -float('inf')), rank, rank_in_action

def read_question(idx, new_system_prompt=False):
    file_name = str(idx).zfill(4)
    img_path = f'question/image/{file_name}.jpg'
    system_text_path = f'question/text/system/{file_name}.txt'
    user_text_path = f'question/text/user/{file_name}.txt'
    
    if not os.path.exists(system_text_path) or not os.path.exists(user_text_path):
        raise FileNotFoundError(f"{file_name}.txt")
    user_content = []
    user_prompt = ''.join(open(user_text_path).readlines())
    if not new_system_prompt:
        system_content = [{"type": "text", "text": ''.join(open(system_text_path).readlines())}]
    else:
        system_prompt, _ = get_system_prompt(with_obs=True, selection=True, with_example=False)
        example_prompt, example_img_url = get_example_prompt(with_image=True, selection=True)
        system_content = [{"type": "text", "text": system_prompt}]
        user_prompt = example_prompt + user_prompt
        for url in example_img_url:
            user_content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
    if os.path.exists(img_path):
        user_content.extend([
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": encode_image(img_path), "detail": "high"}}
        ])
    else:
        user_content.extend([{"type": "text", "text": user_prompt}])
    return system_content, user_content, parse_list(open(user_text_path).read())
        
def main(model='gpt-4o', splitter='\t', force=False, output_file=None):
    assert output_file is not None, "output_file should not be None"
    raw_answer_path = 'answer/answer.txt'
    answer_list = [l.strip().split(splitter) for l in open(raw_answer_path).readlines()]
    answer_list_rank_in_action = [l.strip().split(splitter) for l in open(raw_answer_path).readlines()]
    output_file_rank_in_action = output_file.split('.')[0] + '_rank_in_action.txt'
    data_size = len(answer_list)
    for i in tqdm(range(data_size), ncols=100):
        answer = answer_list[i]
        if not force and len(answer) != 1:
            continue
        system_content, user_content, action_list = read_question(i)
        action_list = [action.split('. ')[0] for action in action_list]
        log_prob, rank, rank_in_action = get_log_prob(system_content, user_content, answer[0], model, action_list)
        prob = np.exp(log_prob)
        answer_list[i].append(str(prob))
        answer_list[i].append(str(rank))
        answer_list_rank_in_action[i].append(str(prob))
        answer_list_rank_in_action[i].append(str(rank_in_action))
    with open(output_file, 'w') as f:
        content = [splitter.join(l) for l in answer_list]
        f.write('\n'.join(content))
    with open(output_file_rank_in_action, 'w') as f:
        content = [splitter.join(l) for l in answer_list_rank_in_action]
        f.write('\n'.join(content))
  
if __name__ == '__main__':
    fail_pair = main(force=True, output_file='answer/answer_sys_6_temp_0.txt')
    