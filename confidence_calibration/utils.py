def parse_list(string):
    sub_string = string[string.find('[')+1:string.find(']')]
    parsed_list = sub_string.split(',')
    parsed_list = [l.strip().replace("'", '') for l in parsed_list]
    return parsed_list
