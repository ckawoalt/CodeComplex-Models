from tqdm import tqdm

import torch
import numpy as np

from sklearn.metrics import confusion_matrix
import argparse


from transformers import ( RobertaTokenizer,
                          AutoTokenizer, 
                          get_linear_schedule_with_warmup,
                          set_seed)
from models import *
from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random

from utils.CodeT5utils import *
from utils.PLBart_utils import *
from utils.comple_utils import * 
from utils.GraphCodeBERT_utils import *

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
            'CodeT5':'Salesforce/codet5-base',}
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def train(args):
 
    set_seed(args)
    tokenizer_type=args.model if args.model != 'comple' else args.submodule
    tokenizer = tokenizers[tokenizer_type].from_pretrained(pretrained_model_name_or_path=model_names[tokenizer_type])

    train_dataset = datasets[args.model](path=args.train_path,tokenizer=tokenizer,args=args,comple=tokenizer_type)
    test_dataset = datasets[args.model](path=args.valid_path,tokenizer=tokenizer,args=args,comple=tokenizer_type)

    model = integrated_model(args)
    
    device=args.device
    # print(device)
    print('Created `train_dataset` with %d examples!'%len(train_dataset))
    print('Created `test_dataset` with %d examples!'%len(test_dataset))

    # Move pytorch dataset into dataloader.
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fns[args.model])
    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))
    valid_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fns[args.model])
    print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))
    datatype=args.train_path
    _model=f'data: {datatype}_{args.model}'+(f'_{args.submodule}' if args.model == 'comple' else '')+f'_pretrain:{args.pretrain}'

    eventid = datetime.now().strftime(f'runs/{_model}-%Y%m-%d%H-%M%S-')

    writer = SummaryWriter(eventid)
    args.max_steps=args.epoch*len(train_dataloader)
    args.warmup_steps=args.max_steps//5

    print('Model loaded')
    if not args.s:
        optimizer = torch.optim.AdamW(model.parameters(),
            lr = args.lr, # - default is 5e-5
            eps = args.adam_epsilon, #  - default is 1e-8.
            weight_decay=args.wd#- default is 1e-2.
            )

    else:
        print("Separated learing rate")
        submodule_params = list(param[1] for param in filter(lambda kv: kv[0].startswith(args.submodule), model.named_parameters()))
        base_params = list(param[1] for param in filter(
            lambda kv: not kv[0].startswith(args.submodule), model.named_parameters()))
        optimizer = torch.optim.AdamW([{"params": submodule_params, "lr": args.sub_lr, "weight_decay": args.sub_wd},
                                       {"params": base_params, "lr":args.lr, "weight_decay": args.wd}
                            ],
                            eps = args.adam_epsilon # - default is 1e-8.
                    )
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    best_loss = np.inf
    best_acc = 0

    for epoch in range(args.epoch):

        total_loss = 0
        val_total_loss = 0

        predictions_labels = []
        true_labels = []

        val_prediction_labels = []
        val_true_labels = []

        model.train()

        for batch in tqdm(train_dataloader, total=len(train_dataloader)):

            model.zero_grad()

            if args.model in ['CodeBERT','PLBART','comple']:
                label = batch['labels'].to(device)
            else:
                label = batch[-1].to(device)

            logits = model(batch)

            loss = criterion(logits, label)

            total_loss += loss.detach().cpu().item()

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Convert these logits to list of predicted labels values.
            true_labels += label.cpu().numpy().flatten().tolist()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
            torch.cuda.empty_cache()
        conf_mat = confusion_matrix(true_labels, predictions_labels)
        acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

        writer.add_scalar('Training loss', total_loss / len(train_dataloader), epoch)
        writer.add_scalar('Training accuracy',  acc, epoch)

        print('Epoch {}, Train loss: {}, accuracy: {} %'.format(epoch, total_loss / len(train_dataloader), acc*100))

        model.eval()

        for batch in tqdm(valid_dataloader, total=len(valid_dataloader)):

            if args.model in ['CodeBERT','PLBART','comple']:
                label = batch['labels'].to(device)
            else:
                label = batch[-1].to(device)

            logits = model(batch)
            loss = criterion(logits, label)

            val_total_loss += loss.detach().item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Convert these logits to list of predicted labels values.
            val_true_labels += label.cpu().numpy().flatten().tolist()
            val_prediction_labels += logits.argmax(axis=-1).flatten().tolist()

        conf_mat = confusion_matrix(val_true_labels, val_prediction_labels)
        acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

        writer.add_scalar('validation loss', val_total_loss / len(valid_dataloader), epoch)
        writer.add_scalar('validation accuracy',  acc, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(model, f'models/best_acc_checkpoint_{_model}'+'.pt')
            print(f'Best acc model saved at accuracy {acc}')
        
        if val_total_loss < best_loss:
            best_loss = val_total_loss
            torch.save(model, f'models/best_loss_checkpoint_{_model}'+'.pt')
            print(f'Best loss model saved at loss {val_total_loss}')
        print('Validation loss: {}, accuracy: {} %'.format(val_total_loss / len(valid_dataloader), acc*100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', required=False, help='train file path',type=str,default='train_p.txt')
    parser.add_argument('--valid_path', required=False, help='valid file path',type=str,default='test_p.txt')

    parser.add_argument('--epoch', required=False, help='number of training epoch',type=int,default=15)
    parser.add_argument('--batch', required=False, help='number of batch size',type=int,default=6)

    parser.add_argument('--model', required=False, help='selelct main model',choices=['CodeBERT','PLBART','GraphCodeBERT','CodeT5','comple'])
    parser.add_argument('--submodule', required=False, help='select submodlue for comple model',choices=['CodeBERT','PLBART','GraphCodeBERT','CodeT5'])

    parser.add_argument('--device', required=False, help='select device for cuda',type=str,default='cuda:0')
    parser.add_argument('--seed', required=False, type=int,default=770)

    parser.add_argument('--max_code_length', required=False, help='max tokenize code length(GraphCodeBERT for 256)',type=int,default=512)
    parser.add_argument('--max_dataflow_length', required=False, help='max tokenize dataflow length(GraphCodeBERT)',type=int,default=64)

    parser.add_argument('--adam_epsilon', required=False, type=float,default=1e-8)
    parser.add_argument('--lr', required=False,help='learning rate' ,type=float,default=2e-5)
    parser.add_argument('--sub_lr', required=False,help='submodule learning rate' ,type=float,default=2e-5)
    parser.add_argument('-wd', required=False,help='weight decay', type=float,default=1e-3)
    parser.add_argument('-sub_wd', required=False,help='submodule weight decay', type=float,default=1e-2)

    parser.add_argument('--pretrain', required=False, action='store_true',help='use pretrain weight for set tranformer and submodule')
    parser.add_argument('--s', action='store_true', help='defer lr(between submodule and transformer)')
    args = parser.parse_args()
    train(args)
