import torch
from torch.utils.data import TensorDataset

class ClassInputFeatures(object):
    def __init__(self,
                 source_ids,
                 label
                 ):
        self.source_ids = source_ids
        self.label = int(label)

def load_classify_data(path, tokenizer,args,comple=None):
    data=open('data/'+path).read().split('\n')[:-1]
    features=[]
    for item in data:
        complextity,code=item.split('\t')
        features.append(convert_class_examples_to_features((code,complextity),tokenizer,args))
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    data = TensorDataset(all_source_ids, all_labels)

    return  data

def convert_class_examples_to_features(item,tokenizer,args):
    source_str,target  = item
    code = tokenizer.encode(source_str, max_length=args.max_code_length, padding='max_length', truncation=True)
    return ClassInputFeatures(code,target)
