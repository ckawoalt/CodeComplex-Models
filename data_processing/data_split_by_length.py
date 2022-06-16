import random
import argparse
from AST2Code import AST2Code_module
import javalang
import os

def augmentation(args):
    file=open(f'data/{args.file_name}').read().split('\n')[:-1]
    datas=[x.split('\t') for x in file]

    codes=[]
    complexity=[]
    for item in datas:
        complexity.append(item[0])
        codes.append(item[1])
    length_dict={'256':[],'512':[], '1024':[],'over':[]}
    code_length={'256':0,'512':0, '1024':0,'over':0}

    for i,code in enumerate(codes):
        codelength=len(list(javalang.tokenizer.tokenize(code)))
        if codelength<256:
            length_dict['256'].append(complexity[i]+'\t'+code)
            code_length['256']+=len(code)
        elif codelength<512:
            length_dict['512'].append(complexity[i]+'\t'+code)
            code_length['512']+=len(code)

        elif codelength<1024:
            length_dict['1024'].append(complexity[i]+'\t'+code)
            code_length['1024']+=len(code)

        else:
            length_dict['over'].append(complexity[i]+'\t'+code)
            code_length['over']+=len(code)

    for k in length_dict.keys():
        print(k,' length :',len(length_dict[k]))
        print(k,' mean :',code_length[k]/len(length_dict[k]),end='\n\n')
        with open(os.path.join('data/length_split', f'{k}_{args.file_name}'), 'w', encoding='utf8') as f:
            f.writelines('\n'.join(length_dict[k]))

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--file_name', required=False, help='probablilty of augmentaion',type=str,default='test_p.txt')
    args = parser.parse_args()
    augmentation(args)