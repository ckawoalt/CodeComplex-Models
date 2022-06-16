from torch.utils.data import Dataset
import torch
class CodeDataset(Dataset):

    def __init__(self, path, tokenizer, args, comple=None):
        # Check max sequence length.
        self.max_sequence_len = tokenizer.max_len if args.max_code_length is None else args.max_code_length

        print('Reading partitions...')
        # Since the labels are defined by folders with data we loop
        # through each label.
        self.tokenizer=tokenizer
        lines=open('data/'+path).read().split('\n')[:-1]
        self.texts = []
        self.labels = []

        for line in lines:
            self.labels.append(int(line.split('\t')[0]))
            self.texts.append(line.split('\t')[1])

        # Number of exmaples.
        self.n_examples = len(self.texts)
        # Use tokenizer on texts. This can take a while.
        print('Finished!\n')

    def __len__(self):
        r"""When used `len` return the number of examples.

        """

        return self.n_examples

    def __getitem__(self, item):
        r"""Given an index return an example from the position.

        Arguments:

          item (:obj:`int`):
              Index position to pick an example to return.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.

        """

        # Use tokenizer on texts. This can take a while.


        return self.texts[item],self.labels[item],self.tokenizer,self.max_sequence_len

def collate_fn(batch):

    _,_, tokenizer, max_sequence_len =batch[0]
    texts=[]
    labels=[]

    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt',  max_length=max_sequence_len)

    inputs.update({'labels':torch.tensor(labels)})
    return inputs