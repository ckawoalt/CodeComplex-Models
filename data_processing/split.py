import random
random.seed(100)
import argparse
from AST2Code import AST2Code_module
import javalang
import os
import json

labels_ids = {'constant':'0', 'linear':'1','logn':'2', 'quadratic':'3','cubic':'4','nlogn':'5' , 'np':'6'}
rev = {v: k for k, v in labels_ids.items()}
from sklearn.model_selection import train_test_split
import re
def delete_import(code):
    import_rule=re.compile('import{1}[^;]*;')
    package_rule=re.compile('package{1}[^;]*;')
    code=import_rule.sub('',code)
    code=package_rule.sub('',code)
    return code

def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "" # so we will return empty to remove the comment
        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, string)
    
def preprocessing(code):

    no_comment_code = remove_comments(code)
    no_comment_code=delete_import(no_comment_code).replace('  ', ' ').strip().replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').strip()
    return no_comment_code

def balance_sample(data,size):
    train={}
    test={}
    item_dict={}
    length_dict={}
    for item in data:
        if item[1] in item_dict.keys():
            item_dict[item[1]].add(item[2])
            length_dict[item[1]]+=1
        else:
            item_dict[item[1]]=set([item[2]])
            length_dict[item[1]]=1
    for comple in item_dict.keys():
        while True:
            train_idx=random.sample(item_dict[comple], int(len(item_dict[comple])*size))
            train[comple]=[]
            test[comple]=[]
            for item in data:
                if item[1]== comple:
                    if item[2] in train_idx:
                        train[comple].append([item[0],item[1]])
                    else:
                        test[comple].append([item[0],item[1]])
            if len(train[comple])/length_dict[comple]<(size*1.1) and len(train[comple])/length_dict[comple]>(size*0.9):
                print(f'{comple} train: {len(train[comple])} test: {len(test[comple])}')
                break

    train_data=[]
    test_data=[]
    for comple in item_dict.keys():
        for tt in train[comple]:
            train_data.append(tt)
        for t in test[comple]:
            test_data.append(t)
    return train_data,test_data
def pick_one_problem(data):
    train={}
    test={}
    item_dict={}
    length_dict={}
    for item in data:
        if item[1] in item_dict.keys():
            item_dict[item[1]].add(item[2])
            length_dict[item[1]]+=1
        else:
            item_dict[item[1]]=set([item[2]])
            length_dict[item[1]]=1

    for comple in item_dict.keys():
        while True:
            train_idx=random.sample(item_dict[comple], 1)
            train[comple]=[]
            test[comple]=[]
            for item in data:
                if item[1]== comple:
                    if item[2] not in train_idx:
                        train[comple].append([item[0],item[1]])
                    else:
                        test[comple].append([item[0],item[1]])
            if len(test[comple])>20:
                print(f'{comple} train: {len(train[comple])} test: {len(test[comple])}')
                break

    train_data=[]
    test_data=[]
    for comple in item_dict.keys():
        for tt in train[comple]:
            train_data.append(tt)
        for t in test[comple]:
            test_data.append(t)
    return train_data,test_data
def split_complexity_equal(data,size):

    p_size=(len(data)/7)*size
    print(f'size:{size}')
    train={}
    test={}
    item_dict={}
    length_dict={}
    for item in data:
        if item[1] in item_dict.keys():
            item_dict[item[1]].add(item[2])
            length_dict[item[1]]+=1
        else:
            item_dict[item[1]]=set([item[2]])
            length_dict[item[1]]=1
    for comple in item_dict.keys():
        while True:
            train_idx=random.sample(item_dict[comple], int(len(item_dict[comple])*size))
            train[comple]=[]
            test[comple]=[]
            for item in data:
                if item[1]== comple:
                    if item[2] in train_idx:
                        train[comple].append([item[0],item[1]])
                    else:
                        test[comple].append([item[0],item[1]])
            if len(train[comple])<(p_size*1.1) and len(train[comple])>(p_size*0.9):
                print(f'{rev[comple]} train: {len(train[comple])} test: {len(test[comple])}')
                break

    train_data=[]
    test_data=[]
    for comple in item_dict.keys():
        for tt in train[comple]:
            train_data.append(tt)
        for t in test[comple]:
            test_data.append(t)
    return train_data,test_data

def delete_error(datas):
    module=AST2Code_module()
    for idx in reversed(range(len(datas))):
        source=datas[idx][0]
        try:
            tree=javalang.parse.parse(source)
            converted=module.AST2Code(tree)
            javalang.parse.parse(converted)
        except:
            del datas[idx]

def print_distribution(data,name):
    dist={}
    for t in data:
        complexity=t.split('\t')[0]
        if complexity in dist.keys():
            dist[complexity]+=1/len(data)
        else:
            dist[complexity]=1/len(data)
    dist=sorted(dist.items())
    tmp_dist={}
    for t in dist:
        tmp_dist[rev[str(int(t[0]))]]=t[1]
    print(f'--------{name}--------')
    for i in tmp_dist:
        print(i,":",round(tmp_dist[i],3))

def split(args):
    datas = []

    with open('data/data.jsonl', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            obj=json.loads(line)
            datas.append([preprocessing(obj['src']), labels_ids[obj['complexity']], obj['problem']])
            
    delete_error(datas)

    if args.type=='p':
        train,test=balance_sample(datas,args.size)
    elif args.type=='o':
        train,test=pick_one_problem(datas)
    elif args.type=='c':
        train,test=split_complexity_equal(datas,args.size)
        
    else:
        codes=[]
        complexity=[]
        for item in datas:
            codes.append(item[0])
            complexity.append(item[1])
        x_train, x_test, y_train, y_test = train_test_split(codes, complexity, test_size=1-args.size, random_state=777, stratify=complexity)
        train=[]
        test=[]
        for idx in range(len(x_train)):
            train.append([x_train[idx],y_train[idx]])
        for idx in range(len(x_test)):
            test.append([x_test[idx],y_test[idx]])
    train_data=[]
    for item in train:
        tmp_code=item[0]
        item[0]=item[1]
        item[1]=tmp_code
        train_data.append('\t'.join(item)+'\n')
    test_data=[]
    for item in test:
        tmp_code=item[0]
        item[0]=item[1]
        item[1]=tmp_code
        test_data.append('\t'.join(item)+'\n')


    train_data=list(set(train_data))
    test_data=list(set(test_data)-set(train_data))
    print('train:',len(train_data),'\n','test:',len(test_data))
    print('train:',round(len(train_data)/(len(train_data)+len(test_data)),3),'\n','test:',round(len(test_data)/(len(train_data)+len(test_data)),3))

    with open(os.path.join('data', f'train_{args.type}.txt'), 'w', encoding='utf8') as f:
        f.writelines(train_data)
    with open(os.path.join('data', f'test_{args.type}.txt'), 'w', encoding='utf8') as f:
        f.writelines(test_data)      

    print_distribution(train_data,'train')
    print_distribution(test_data,'test')

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--type', required=True, help='probablilty of augmentaion',choices=['r','p'])
    parser.add_argument('--size', required=False, help='learning rate',type=float,default=0.8)
    args = parser.parse_args()
    split(args)