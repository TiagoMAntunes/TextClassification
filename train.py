import pytreebank
import numpy as np
import torch
import sys
import argparse
from glove_reader import GloveReader
from gensim.utils import tokenize
import itertools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='./trees/train.txt')
    # parser.add_argument('')
    
    args = parser.parse_args()
    
    #loads glove embeddings
    glove = GloveReader()
    
    #load dataset
    data = pytreebank.import_tree_corpus(args.data_dir)
    data = list(map(lambda x: x.to_labeled_lines()[0], data))
    
    # tokenizes and filters those words that exist in the dictionary for each example
    data = list(map(lambda x: (x[0], list(filter(lambda x: x in glove.words2idx, tokenize(x[1], lower=True)))), data))

    # transforms words into numbers
    data = list(map(lambda x: (x[0],*list(map(lambda y: glove.words2idx[y], x[1]))), data))

    # pad data and transform tensor
    data = torch.tensor(list(itertools.zip_longest(*data, fillvalue=0))).T