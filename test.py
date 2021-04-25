import pickle

import numpy as np
import pandas
import pytreebank
import torch
import torch.nn as nn
from gensim.utils import tokenize

from dataloader import SST
from glove_reader import GloveReader
from model import RNN
from util import load_data


glove = GloveReader()


test_data = SST(*load_data('./trees/test.txt', glove))
train_data = SST(*load_data('./trees/train.txt', glove))
dev_data = SST(*load_data('./trees/dev.txt', glove))

from itertools import product

configuration = {
    'dropout': [0, 0.2],
    'hidden_size' : [256, 512],
    'n_layers': [1, 3],
    'embeddings': [glove.embeddings, None]
}
model_params = list(product(*configuration.values()))


def validate(model, dataset):
    dev = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=2, shuffle=False)
    
    model.eval()
    
    loss_fn = torch.nn.CrossEntropyLoss()

    total_count = 0
    num_correct = 0
    tot_loss = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in dev:
        data, labels, mask = batch
        data = data.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        output = model(data,mask)
        tot_loss += loss_fn(output, labels).item()
        
        pred = torch.argmax(output, 1)
        num_correct += (pred == labels).sum().item()
        total_count += pred.size(0)
        del data
        del labels
        del mask

    model.train()
    return num_correct / total_count, tot_loss

import pickle

results = []
device = "cuda" if torch.cuda.is_available() else "cpu"
for params in model_params:
    info = f'{params[0]}_{params[1]}_{params[2]}_{"glove" if params[3] is not None else "default"}'
    model_name = f'model_{info}'
    data_name = f'data_{info}'

    with open(data_name, 'rb') as f:
        data = pickle.load(f)
    
    m = RNN(300, params[2], params[1], 5, pretrained_embeddings=params[3], dropout=params[0])

    m.load_state_dict(torch.load(model_name, map_location=device))

    test_acc, test_loss = validate(m, test_data)
    train_acc, train_loss = validate(m, train_data)
    dev_acc, dev_loss = validate(m, dev_data)
    results.append({'test_accuracy':test_acc, 'train_acc': train_acc, 'dev_acc': dev_acc, 'test_loss': test_loss, 'train_loss':train_loss, 'dev_loss': dev_loss, 'name': info})

  

with open ('results.pickle', 'wb') as f:
    pickle.dump(results, f)


for i,res in enumerate(results):
    dropout, hidden_size, n_layers, embeddings = res['name'].split('_')
    print(f'{dropout} & {hidden_size} & {n_layers} & {embeddings} & {round(res["test_accuracy"], 4)} & {round(res["test_loss"], 4)} & {round(res["train_acc"],4)} & {round(res["dev_acc"], 4)}\\\\')
    if (i+1) % 2 == 0:
        print('\\hline')




