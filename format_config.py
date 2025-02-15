import yaml


def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
        
def format(config):
    formatted_config = ""
    # for each column in the config file, concat the instructions
    for key, value in config.items():
        for key2, value2 in value.items():
            formatted_config += f"{key2}. Instruction - {value2['instruction']}\n"
    formatted_config += "\n"
    for key, value in config.items():
        for key2, value2 in value.items():
            ans = value2['answer']
            for i, a in enumerate(ans):
                formatted_config += f"{i+1}. {a}\n"
            formatted_config += "\n"
    with open('format_config.txt', 'w') as f:
        f.write(formatted_config)


if __name__ == '__main__':
    config = load_config('src/config/obstacles.yaml')
    format(config)