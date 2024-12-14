import os
from openai import OpenAI
import base64
import mimetypes
from tqdm import tqdm

openai_client = OpenAI()
def encode_image(image_path: str):
    """Encodes an image to base64 and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"

def get_log_prob(system_content, user_content, answer, model):
      
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
    return top_logprobs.get(answer, None)

def read_question(idx):
    file_name = str(idx).zfill(4)
    img_path = f'question/image/{file_name}.jpg'
    system_text_path = f'question/text/system/{file_name}.txt'
    user_text_path = f'question/text/user/{file_name}.txt'
    
    if not os.path.exists(system_text_path) or not os.path.exists(user_text_path):
        raise FileNotFoundError(f"{file_name}.txt")
    system_content = [{"type": "text", "text": ''.join(open(system_text_path).readlines())}]
    if os.path.exists(img_path):
        user_content = [
            {"type": "text", "text": ''.join(open(user_text_path).readlines())},
            {"type": "image_url", "image_url": {"url": encode_image(img_path), "detail": "high"}}
        ]
    else:
        user_content = [{"type": "text", "text": ''.join(open(user_text_path).readlines())}]
    return system_content, user_content
        
def main(model='gpt-4o'):
    answer_list = [l.strip().split() for l in open('answer.txt').readlines()]
    data_size = len(answer_list)
    fail_pair = []
    for i in tqdm(range(data_size), ncols=100):
        answer = answer_list[i]
        if len(answer) != 1:
            continue
        system_content, user_content = read_question(i)
        log_prob = get_log_prob(system_content, user_content, answer[0], model)
        if log_prob:
            answer_list[i].append(str(log_prob))
        else:
            fail_pair.append(i)
    with open('answer.txt', 'w') as f:
        content = [' '.join(l) for l in answer_list]
        f.write('\n'.join(content))
    return fail_pair
        
if __name__ == '__main__':
    fail_pair = main()
    print(fail_pair)