answer_list = open('answer.txt').readlines()
open('answer.txt', 'w').write('\n'.join([a.split()[0] for a in answer_list]))