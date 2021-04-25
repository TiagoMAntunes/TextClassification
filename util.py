
import numpy as np
import pandas as pd
import pytreebank
import torch
from gensim.utils import tokenize

def transform_and_pad(data):
    """
        Given the input dataset, returns three tensors of the padded data (on the left!)
        Returns:
            content - Tensor Nxd
            labels  - Tensor Nx1
            mask    - Tensor Nxd (binary)
        
    """
    max_len = max(map(lambda x: len(x[1:]), data))
    labels = torch.tensor(list(map(lambda x: x[0], data)))
    data = list(map(lambda x: x[1:], data))
    content = torch.tensor([(0,)*(max_len - len(x)) + x for x in data])
    content_mask = torch.tensor([(0,)*(max_len - len(x)) + (1,) * len(x) for x in data])
    return content, labels, content_mask


def load_data(path, glove):
    data = pytreebank.import_tree_corpus(path)
    data = list(map(lambda x: x.to_labeled_lines()[0], data))

    # tokenizes and filters those words that exist in the dictionary for each example
    data = list(map(lambda x: (x[0], list(filter(lambda x: x in glove.words2idx, tokenize(x[1], lower=True)))), data))

    # transforms words into numbers
    data = list(map(lambda x: (x[0],*list(map(lambda y: glove.words2idx[y], x[1]))), data))

    # pad data and transform tensor
    content, labels, mask = transform_and_pad(data)
    del data
    
    return content, labels, mask


