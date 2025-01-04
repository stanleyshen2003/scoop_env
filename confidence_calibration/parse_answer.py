import os
from utils import parse_list
def main():
    with open('answer.txt', 'r') as f:
        answer_list = [l.strip().split()[0] for l in f.readlines()]
    new_answer_list = []
    for i, answer in enumerate(answer_list):
        user_prompt_path = f'question/text/user/{str(i).zfill(4)}.txt'
        action_list = parse_list(open(user_prompt_path).read())
        action = action_list[ord(answer) - ord('A')].replace(f'{answer}. ', '')
        new_answer_list.append([answer, action])
    with open('answer.txt', 'w') as f:
        f.write('\n'.join(['\t'.join(l) for l in new_answer_list]))
        
        
if __name__ == '__main__':  
    main()