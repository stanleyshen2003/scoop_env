import yaml


def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

def load_urls_from_txt(url_path):
    with open(url_path, 'r') as f:
        urls = f.readlines()
    return urls
        
def format(config, urls):
    formatted_config = ""
    # for each column in the config file, concat the instructions
    for key, value in config.items():
        for key2, value2 in value.items():
            formatted_config += f"{key2}. **Instruction - {value2['instruction']}**\n"
            ans = value2['answer']
            formatted_config += "\t:::spoiler Expected sequence\n"
            for i, a in enumerate(ans):
                formatted_config += f"\t{i+1}. {a}\n"
            formatted_config += "\t:::\n"
            formatted_config += "\t" + urls[key2-1] + "\n"
    formatted_config += "\n"
    
    with open('format_config.txt', 'w') as f:
        f.write(formatted_config)


if __name__ == '__main__':
    config = load_config('src/config/final_task/amount_ambiguity.yaml')
    urls = load_urls_from_txt('format_urls.txt')
    format(config, urls)