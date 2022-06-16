import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel,AdamW,RobertaTokenizer,AutoConfig,RobertaForSequenceClassification,RobertaConfig
from models import GraphCodeBERT_model
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'set_transformer'))
from models import SetTransformer_encoder
from set_transformer.blocks import InducedSetAttentionBlock
import argparse

from tqdm import tqdm
from torch.utils.data import  Dataset

tokenizers={'CodeBERT':AutoTokenizer,
            'GraphCodeBERT':RobertaTokenizer}

model_names={'CodeBERT':'microsoft/codebert-base-mlm',
            'GraphCodeBERT':'microsoft/graphcodebert-base',}

BCE_loss_fn=nn.BCEWithLogitsLoss()
MSE_loss_fn=nn.MSELoss()
MLMLoss=nn.NLLLoss(ignore_index=-100)

class CompleDataset(Dataset):
    
    def __init__(self, path,tokenizer,args,comple='CodeBERT'):
        self.data=[]
        token_funs={'CodeBERT':get_CodeBERT_token}
        with open(os.path.join('data', path),"rb") as fr:
            lines = pickle.load(fr)

        if 'test' in path:
            lines=lines[:4000]
            
        for line in tqdm(lines):
            Connect=line['Connect'][:args.max_method_length,:args.max_method_length]
            code=line['Codes'][:args.max_method_length]
            is_recursive=line['Recursive'][:args.max_method_length]
            depth=line['Depth'][:args.max_method_length]
            is_sort=line['CallSort'][:args.max_method_length]
            is_hash_map=line['HashMap'][:args.max_method_length]
            is_hash_set=line['HashSet'][:args.max_method_length]
            num_prams=line['NumParams'][:args.max_method_length]

            func_tokens=[]
            MLM_targets=[]
            for fun in code: # for each class code
                tokenized = token_funs[comple](fun,tokenizer,args)
                if args.model=='CodeBERT':
                    _input,MLM=mask_tokens(tokenized['input_ids'],tokenizer)
                    func_tokens.append(_input)
                    MLM_targets.append(MLM)
                else:
                    _input,MLM=mask_tokens(tokenized[0].unsqueeze(0),tokenizer)
                    tokenized[0]=_input.squeeze(0)
                    func_tokens.append(tokenized)
                    MLM_targets.append(MLM)

            self.data.append({'input':func_tokens,'Connect':Connect,'MLM':torch.cat(MLM_targets,dim=1),\
                'Recursive':is_recursive,'Depth':depth,'CallSort':is_sort,'HashMap':is_hash_map,'HashSet':is_hash_set,'NumParams':num_prams})

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        
        return self.data[item]

def get_CodeBERT_token(code,tokenizer,args):
    tokenized = tokenizer(code, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt',  max_length=512)
    return tokenized

def collate_fn_level(batch):
    inputs={}
    total_tokens=[]
    Connect=[]
    is_recursive=[]
    is_sort=[]
    is_hash_map=[]
    is_hash_set=[]
    depth=[]
    MLM=[]
    num_prams=[]
    for item in batch:

        total_tokens.append(item['input'])
        Connect.append(item['Connect'])
        is_recursive.append(item['Recursive'])
        is_sort.append(item['CallSort'])
        is_hash_map.append(item['HashMap'])
        is_hash_set.append(item['HashSet'])
        depth.append(item['Depth'])
        MLM.append(item['MLM'])
        num_prams.append(item['NumParams'])

    inputs['input_ids'] = total_tokens
    MLM=torch.cat(MLM,dim=1).view(1,-1)
    inputs.update({'Connect':torch.tensor(Connect),'Recursive':torch.tensor(is_recursive),'CallSort':torch.tensor(is_sort),\
        'HashMap':torch.tensor(is_hash_map),'HashSet':torch.tensor(is_hash_set),'Depth':torch.tensor(depth),'MLM':MLM,'NumParams':torch.tensor(num_prams)})

    return inputs

class SetTransformer_encoder(nn.Module):
    
    def __init__(self, in_dimension):
        """
        Arguments:
            in_dimension: an integer.
            out_dimension: an integer.
        """
        super().__init__()

        d = 768
        m = 16  # number of inducing points
        h = 4  # number of heads

        self.embed = nn.Sequential(
            nn.Linear(in_dimension, d),
            nn.ReLU(inplace=True)
        )
        self.encoder = nn.Sequential(
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, in_dimension].
        Returns:
            a float tensor with shape [b, out_dimension].
        """

        x = self.embed(x)  # shape [b, n, d]
        x = self.encoder(x)  # shape [b, n, d]

        return x

class RFF(nn.Module):
    """
    Row-wise FeedForward layers.
    """
    def __init__(self, d):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)

class Model(nn.Module):
    def __init__(self,args):
        super().__init__()
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_names[args.model])

        if args.model == 'CodeBERT':
            self.generate_CodeBERT()
        else:
            self.generate_GraphCodeBERT()
        self.device=args.device
        bert_hidden=768
        transformer_hidden=768

        self.transformer_encoder=SetTransformer_encoder(bert_hidden)

        self.connect=nn.Linear(transformer_hidden,args.max_method_length)
        self.vocab_size=model_config.vocab_size
        self.recusive=nn.Linear(transformer_hidden,1)
        self.sort=nn.Linear(transformer_hidden,1)
        self.hash_map=nn.Linear(transformer_hidden,1)
        self.hash_set=nn.Linear(transformer_hidden,1)
        self.depth=nn.Linear(transformer_hidden,1)
        self.MLM=nn.Linear(transformer_hidden,model_config.vocab_size)
        self.num_prams=nn.Linear(transformer_hidden,1)

        self.max_method_length=args.max_method_length
        self.softmax = nn.LogSoftmax(dim=-1)
        self.model=args.model
        
    def forward(self,x):
        batch_size = len(x['input_ids'])

        is_recursive=[]
        is_sort=[]
        is_hash_map=[]
        is_hash_set=[]
        depth=[]
        MLM=[]
        num_prams=[]

        batch_embeddings=[]
        for b in range(batch_size):
            method_embeddings=[]
            for cla in x['input_ids'][b]:
                if self.model =='CodeBERT':
                    method_embed=self.get_CodeBERT_embedding(cla)
                    method_embeddings.append(method_embed['pooler_output'])
                    MLM.append(self.softmax(self.MLM(method_embed['last_hidden_state'])))
                else:
                    method_embed=self.get_GraphCodeBERT_embedding(cla)
                    method_embeddings.append(method_embed[:,0,:])
                    MLM.append(self.softmax(self.MLM(method_embed)))
    
            method_embeddings=torch.stack(method_embeddings,dim=1)[0]
            
            is_recursive.append(self.recusive(method_embeddings))
            is_sort.append(self.sort(method_embeddings))
            is_hash_map.append(self.hash_map(method_embeddings))
            is_hash_set.append(self.hash_set(method_embeddings))
            depth.append(self.depth(method_embeddings))
            
            num_prams.append(self.num_prams(method_embeddings))
        
            batch_embeddings.append(method_embeddings)

        batch_embeddings=torch.stack(batch_embeddings,dim=0)
        output=self.transformer_encoder(batch_embeddings)
        output=self.connect(output)

        return {'Connect':output.reshape(batch_size,-1),
        'Recursive':torch.stack(is_recursive,dim=0).reshape(batch_size,-1),
        'CallSort':torch.stack(is_sort,dim=0).reshape(batch_size,-1),
        'HashMap':torch.stack(is_hash_map,dim=0).reshape(batch_size,-1),
        'HashSet':torch.stack(is_hash_set,dim=0).reshape(batch_size,-1),
        'Depth':torch.stack(depth,dim=0).reshape(batch_size,-1),
        'MLM':torch.cat(MLM,dim=1).reshape(1,-1,self.vocab_size),
        'NumParams':torch.stack(num_prams,dim=0).reshape(batch_size,-1)}


    def generate_CodeBERT(self):
            
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path='microsoft/codebert-base-mlm')
        self.CodeBERT = AutoModel.from_pretrained(pretrained_model_name_or_path='microsoft/codebert-base-mlm', config=model_config)

    def generate_GraphCodeBERT(self):
        model_path="microsoft/graphcodebert-base"
        config = RobertaConfig.from_pretrained(model_path,num_labels=7)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        encoder = RobertaForSequenceClassification.from_pretrained(model_path,config=config)    
        self.GraphCodeBERT=GraphCodeBERT_model(encoder,config,tokenizer,None)

    def get_CodeBERT_embedding(self,x):
        output=self.CodeBERT(x.to(self.device))
        return output

    def get_GraphCodeBERT_embedding(self,x):
        (inputs_ids,position_idx,attn_mask)=[item.to(self.device)  for item in x]

        output=self.GraphCodeBERT(inputs_ids,position_idx,attn_mask)
        return output[0]

def get_loss(pred,target,device):

    Connect_loss=BCE_loss_fn(pred['Connect'],target['Connect'].to(device))
    Recursive_loss=BCE_loss_fn(pred['Recursive'],target['Recursive'].float().to(device))
    CallSort_loss=BCE_loss_fn(pred['CallSort'],target['CallSort'].float().to(device))
    HashMap_loss=BCE_loss_fn(pred['HashMap'],target['HashMap'].float().to(device))
    HashSet_loss=BCE_loss_fn(pred['HashSet'],target['HashSet'].float().to(device))
    Depth_loss=MSE_loss_fn(pred['Depth'],target['Depth'].float().to(device))
    MLM_loss=MLMLoss(pred['MLM'].squeeze(0),target['MLM'].squeeze(0).to(device))
    NumParams_loss=MSE_loss_fn(pred['NumParams'],target['NumParams'].float().to(device))

    total_loss=Connect_loss+Recursive_loss+CallSort_loss+HashMap_loss+HashSet_loss+Depth_loss+MLM_loss+NumParams_loss

    return {'total_loss':total_loss,'Connect_loss':Connect_loss,'Recursive_loss':Recursive_loss,\
        'CallSort_loss':CallSort_loss,'HashMap_loss':HashMap_loss,'HashSet_loss':HashSet_loss,\
        'Depth_loss':Depth_loss,'MLM_loss':MLM_loss,'NumParams_loss':NumParams_loss}

def mask_tokens( inputs: torch.Tensor,tokenizer,mlm_probability=0.15, pad=True):
    labels = inputs.clone()

    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
      tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
      padding_mask = labels.eq(tokenizer.pad_token_id)
      probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]


    return inputs, labels

def train(args):

    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_names[args.model])

    train_set=CompleDataset(args.train_path,tokenizer,args,comple=args.model)
    valid_set=CompleDataset(args.valid_path,tokenizer,args,comple=args.model)

    train_dataloader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn_level)
    valid_dataloader=DataLoader(valid_set,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn_level)

    eventid = datetime.now().strftime(f'runs/%Y%m-%d%H-%M%S-')
    writter=SummaryWriter(eventid)
    device = torch.device(args.device)

    model=Model(args)
    model.to(device)
    
    optimizer = AdamW(model.parameters(),
        lr = 5e-6, # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
        weight_decay=0.003
        )

    saved=0
    for epoch in range(args.epoch):
        model.train()

        total_loss_sum,Connect_loss_sum,Recursive_loss_sum,CallSort_loss_sum,HashMap_loss_sum,HashSet_loss_sum,Depth_loss_sum,MLM_loss_sum,NumParams_loss_sum=\
            0,0,0,0,0,0,0,0,0
        cur_step = 0

        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            # label = batch['labels'].view(batch['labels'].size(0),args.max_method_length**2).to(device)
            target=batch
            target['Connect']=target['Connect'].view(batch['Connect'].size(0),args.max_method_length**2)
            y = model(batch)
            optimizer.zero_grad()
            loss_dict=get_loss(y,target,device)

            loss_dict['total_loss'].requires_grad_(True)
            loss_dict['total_loss'].backward()
  
            optimizer.step()
            
            cur_step+=1

            total_loss_sum+=loss_dict['total_loss'].detach().cpu().item()
            Connect_loss_sum+=loss_dict['Connect_loss'].detach().cpu().item()
            Recursive_loss_sum+=loss_dict['Recursive_loss'].detach().cpu().item()
            CallSort_loss_sum+=loss_dict['CallSort_loss'].detach().cpu().item()
            HashMap_loss_sum+=loss_dict['HashMap_loss'].detach().cpu().item()
            HashSet_loss_sum+=loss_dict['HashSet_loss'].detach().cpu().item()
            Depth_loss_sum+=loss_dict['Depth_loss'].detach().cpu().item()
            MLM_loss_sum+=loss_dict['MLM_loss'].detach().cpu().item()
            NumParams_loss_sum+=loss_dict['NumParams_loss'].detach().cpu().item()

            if cur_step % args.save_step == 0:
                writter.add_scalar('train/total_loss',total_loss_sum/args.save_step,)
                writter.add_scalar('train/Connect_loss',Connect_loss_sum/args.save_step,saved)
                writter.add_scalar('train/Recursive_loss',Recursive_loss_sum/args.save_step,saved)
                writter.add_scalar('train/CallSort_loss',CallSort_loss_sum/args.save_step,saved)
                writter.add_scalar('train/HashMap_loss',HashMap_loss_sum/args.save_step,saved)
                writter.add_scalar('train/HashSet_loss',HashSet_loss_sum/args.save_step,saved)
                writter.add_scalar('train/Depth_loss',Depth_loss_sum/args.save_step,saved)
                writter.add_scalar('train/MLM_loss',MLM_loss_sum/args.save_step,saved)
                writter.add_scalar('train/NumParams_loss',NumParams_loss_sum/args.save_step,saved)
                
                total_loss_sum,Connect_loss_sum,Recursive_loss_sum,CallSort_loss_sum,HashMap_loss_sum,HashSet_loss_sum,Depth_loss_sum,MLM_loss_sum,NumParams_loss_sum=\
                    0,0,0,0,0,0,0,0,0
                evaluate(model,writter,valid_dataloader,saved,device)
                saved+=1
                torch.save(model.transformer_encoder, f'pretrain_save/transformer_encoder_train_epoch_{args.model}_{epoch}_{cur_step}'+'.pt')
                if args.model=='CodeBERT':
                    torch.save(model.CodeBERT, f'pretrain_save/transformer_CodeBERT_train_epoch_{args.model}_{epoch}_{cur_step}'+'.pt')
                else:
                    torch.save(model.GraphCodeBERT, f'pretrain_save/transformer_GraphCodeBERT_train_epoch_{args.model}_{epoch}_{cur_step}'+'.pt')
        # torch.cuda.empty_cache()

def evaluate(model,writter,valid_dataloader,epoch,device):
    model.eval()
    total_loss_sum,Connect_loss_sum,Recursive_loss_sum,CallSort_loss_sum,HashMap_loss_sum,HashSet_loss_sum,Depth_loss_sum,MLM_loss_sum,NumParams_loss_sum=\
        0,0,0,0,0,0,0,0,0
    cur_step=0
    for batch in tqdm(valid_dataloader, total=len(valid_dataloader)):
        target=batch
        target['Connect']=target['Connect'].view(batch['Connect'].size(0),args.max_method_length**2)
        with torch.no_grad():
            y=model(batch)
        loss_dict=get_loss(y,target,device)
        cur_step+=1

        total_loss_sum+=loss_dict['total_loss'].detach().cpu().item()
        Connect_loss_sum+=loss_dict['Connect_loss'].detach().cpu().item()
        Recursive_loss_sum+=loss_dict['Recursive_loss'].detach().cpu().item()
        CallSort_loss_sum+=loss_dict['CallSort_loss'].detach().cpu().item()
        HashMap_loss_sum+=loss_dict['HashMap_loss'].detach().cpu().item()
        HashSet_loss_sum+=loss_dict['HashSet_loss'].detach().cpu().item()
        Depth_loss_sum+=loss_dict['Depth_loss'].detach().cpu().item()
        MLM_loss_sum+=loss_dict['MLM_loss'].detach().cpu().item()
        NumParams_loss_sum+=loss_dict['NumParams_loss'].detach().cpu().item()

    writter.add_scalar('valid/total_loss',total_loss_sum/cur_step,epoch)
    writter.add_scalar('valid/Connect_loss',Connect_loss_sum/cur_step,epoch)
    writter.add_scalar('valid/Recursive_loss',Recursive_loss_sum/cur_step,epoch)
    writter.add_scalar('valid/CallSort_loss',CallSort_loss_sum/cur_step,epoch)
    writter.add_scalar('valid/HashMap_loss',HashMap_loss_sum/cur_step,epoch)
    writter.add_scalar('valid/HashSet_loss',HashSet_loss_sum/cur_step,epoch)
    writter.add_scalar('valid/Depth_loss',Depth_loss_sum/cur_step,epoch)
    writter.add_scalar('valid/MLM_loss',MLM_loss_sum/cur_step,epoch)
    writter.add_scalar('valid/NumParams_loss',NumParams_loss_sum/cur_step,epoch)

    model.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--train_path', required=False, help='train file path',type=str,default='class_train.pickle')
    parser.add_argument('--valid_path', required=False, help='test file path',type=str,default='class_test.pickle')

    parser.add_argument('--epoch', required=False, help='number of training epoch',type=int,default=4)
    parser.add_argument('--batch_size', required=False, help='number of batch size',type=int,default=6)
    parser.add_argument('--model', required=False, help='selelct main model',choices=['CodeBERT','GraphCodeBERT'],default='CodeBERT')
    parser.add_argument('--save_step', required=False, help='model save step',type=int,default=3)

    parser.add_argument('--device', required=False, help='select device for cuda',type=str,default='cuda:0')

    parser.add_argument('--max_code_length', required=False, help='maximum code length',type=int,default=512)
    parser.add_argument('--max_method_length', required=False, help='maximum method length (<20)',type=int,default=10)

    parser.add_argument('--adam_epsilon', required=False, type=float,default=1e-8)
    parser.add_argument('-lr','--learning_rate', required=False, type=float,default=2e-5)
    parser.add_argument('-wd','--weight_decay', required=False, type=float,default=0.0)

    args = parser.parse_args()
    train(args)