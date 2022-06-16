from tqdm import tqdm
import os
import torch
import numpy as np

from sklearn.metrics import confusion_matrix
import argparse

from transformers import ( RobertaTokenizer,               
                          AutoTokenizer,
                          set_seed,
                          )
from models import *
from torch.utils.data import  DataLoader
import random

from utils.CodeT5utils import *
from utils.PLBart_utils import *
from utils.comple_utils import * 
from utils.GraphCodeBERT_utils import *
from models import integrated_model
labels_ids = {'constant':'0', 'linear':'1','logn':'2', 'quadratic':'3','cubic':'4','nlogn':'5' , 'np':'6'}
# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)

datasets= { 'CodeBERT':CodeDataset,
            'PLBART':CodeDataset,
            'GraphCodeBERT':TextDataset,
            'CodeT5':load_classify_data,
            'comple':CompleDataset}

collate_fns={'CodeBERT':collate_fn,
            'PLBART':collate_fn,
            'GraphCodeBERT':None,
            'CodeT5':None,
            'comple':collate_fn_level_transformer}

tokenizers={'CodeBERT':AutoTokenizer,
            'PLBART':AutoTokenizer,
            'GraphCodeBERT':RobertaTokenizer,
            'CodeT5':RobertaTokenizer,}   

model_names={'CodeBERT':'microsoft/codebert-base-mlm',
            'PLBART':'uclanlp/plbart-base',
            'GraphCodeBERT':'microsoft/graphcodebert-base',
            'CodeT5':'Salesforce/codet5-base'}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def train(args):
    set_seed(args)
    tokenizer_type=args.model if args.model != 'comple' else args.submodule
    tokenizer = tokenizers[tokenizer_type].from_pretrained(pretrained_model_name_or_path=model_names[tokenizer_type])
    test_dataset = datasets[args.model](path=args.valid_path,tokenizer=tokenizer,args=args,comple=tokenizer_type)
    model = integrated_model(args)

    if args.model =='comple':
        if args.pretrain:
            model.load_state_dict(torch.load(f'saved_model/comple_{args.submodule}_pretrain.pt'))
        else:
            model.load_state_dict(torch.load(f'saved_model/comple_{args.submodule}.pt'))
    else:    
        model.load_state_dict(torch.load(f'saved_model/{args.model}.pt'))

    model.to(args.device)

    device=args.device
    print('Created `test_dataset` with %d examples!'%len(test_dataset))

    # Move pytorch dataset into dataloader.
    # random_probs parameter for augmentation. if random_probs == 0 then no augmentation.
    valid_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fns[args.model])
    print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))
    print('Model loaded')

    model = model.to(device)
    
    val_prediction_labels = []
    val_true_labels = []

    model.eval()

    for batch in tqdm(valid_dataloader, total=len(valid_dataloader)):

        if args.model in ['CodeBERT','PLBART','comple']:
            label = batch['labels'].to(device)
        else:
            label = batch[-1].to(device)

        logits = model(batch)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Convert these logits to list of predicted labels values.
        val_true_labels += label.cpu().numpy().flatten().tolist()
        val_prediction_labels += logits.argmax(axis=-1).flatten().tolist()

    conf_mat = confusion_matrix(val_true_labels, val_prediction_labels)
    acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

    
    return 'accuracy: {} %'.format( round(acc*100,2))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', required=False, help='valid file path',type=str,default='test_p.txt')

    parser.add_argument('--epoch', required=False, help='number of training epoch',type=int,default=15)
    parser.add_argument('--batch', required=False, help='number of batch size',type=int,default=6)

    parser.add_argument('--model', required=False, help='selelct main model',choices=['CodeBERT','PLBART','GraphCodeBERT','CodeT5','comple'])
    parser.add_argument('--submodule', required=False, help='select submodlue for comple model',choices=['CodeBERT','PLBART','GraphCodeBERT','CodeT5'])

    parser.add_argument('--device', required=False, help='select device for cuda',type=str,default='cuda:0')
    parser.add_argument('--seed', required=False, type=int,default=770)

    parser.add_argument('--max_code_length', required=False, help='max tokenize code length(GraphCodeBERT for 256)',type=int,default=512)
    parser.add_argument('--max_dataflow_length', required=False, help='max tokenize dataflow length(GraphCodeBERT)',type=int,default=64)

    parser.add_argument('--pretrain', required=False, action='store_true',help='use pretrain weight for set tranformer and submodule')

    args = parser.parse_args()
    result=[]
    result.append(args.test_path+' : '+train(args))

    for i in os.listdir('data/length_split/'):
        args.test_path='length_split/'+i
        result.append(i+' : '+train(args))
    for i in os.listdir('data/complexity_split/'):
        args.test_path='complexity_split/'+i
        result.append(i+' : '+train(args))
    for i in result:
        print(i)
