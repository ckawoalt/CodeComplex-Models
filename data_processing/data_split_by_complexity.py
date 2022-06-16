import argparse
import os

def augmentation(args):
    file=open(f'data/{args.file_name}').read().split('\n')[:-1]
    datas=[x.split('\t') for x in file]

    codes=[]
    complexity=[]
    for item in datas:
        complexity.append(item[0])
        codes.append(item[1])
    length_dict={'constant':[],'logn':[], 'linear':[],'quadratic':[],'nlogn':[],'np':[],'cubic':[]}
    for i,code in enumerate(codes):
        if complexity[i]=='0':
            length_dict['constant'].append(complexity[i]+'\t'+code)
        elif complexity[i]=='1':
            length_dict['linear'].append(complexity[i]+'\t'+code)
        elif complexity[i]=='2':
            length_dict['logn'].append(complexity[i]+'\t'+code)
        elif complexity[i]=='3':
            length_dict['quadratic'].append(complexity[i]+'\t'+code)
        elif complexity[i]=='4':
            length_dict['cubic'].append(complexity[i]+'\t'+code)
        elif complexity[i]=='5':
            length_dict['nlogn'].append(complexity[i]+'\t'+code)
        elif complexity[i]=='6':
            length_dict['np'].append(complexity[i]+'\t'+code)

    for k in length_dict.keys():
        print(k,' length :',len(length_dict[k]))
        with open(os.path.join('data/complexity_split', f'{k}_{args.file_name}'), 'w', encoding='utf8') as f:
            f.writelines('\n'.join(length_dict[k]))

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--file_name', required=False, help='probablilty of augmentaion',type=str,default='test_p.txt')
    args = parser.parse_args()
    augmentation(args)