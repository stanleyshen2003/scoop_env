import yaml
def read_yaml(file_path, env_type='hard', env_num=1):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
config = read_yaml('config.yaml')
new_config = {}
for difficulty, sub_config in config.items():
    for idx, task_config in sub_config.items():
        new_containers_config = []
        for container_config in task_config['containers']:
            new_container_config = {
                'x': container_config['x'],
                'y': container_config['y'],
                'type': container_config['type'],
                'color': container_config['container_color'],
                'food':{
                    'type': container_config['food'],
                    'color': container_config.get('color', ['green']),
                    'amount': container_config.get('amount', 1200),
                    'position': 'center-center'
                }   
            }
            new_containers_config.append(new_container_config)
        task_config['containers'] = new_containers_config
        sub_config[idx] = task_config
    config[difficulty] = sub_config
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False, default_flow_style=None)
print(config)