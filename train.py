import pytreebank
import numpy as np
import torch
import sys
import argparse
from glove_reader import GloveReader
from gensim.utils import tokenize
import itertools
from model import RNN
import time

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

def train(model, data, mask, labels, max_epochs=50):

    losses = []
    accs = []

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(max_epochs):
        print("-" * 10 , "EPOCH ", epoch,  "-"*10)
        
        num_correct = 0.0
        total_count = 0.0

        start = time.time()
        # TODO batched?

        optimizer.zero_grad()

        output = model(data, mask)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        
        end = time.time()
        
        pred = torch.argmax(output, 1)
        num_correct += (pred == labels).sum().item()
        total_count += pred.size(0)
        losses.append(loss.item())
        accs.append(num_correct / total_count)

        print(f'Loss={losses[-1]}, Accuracy={accs[-1]}, took {end - start}s')



def main():
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
    content, labels, mask = transform_and_pad(data)
    del data

    m  = RNN(300, 2, 128, 5)
    # res = m(content, mask)

    train(m, content, mask, labels)


if __name__ == '__main__':
    main()