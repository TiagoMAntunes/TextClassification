import argparse
import itertools
import sys
import time

import numpy as np
import pandas as pd
import pytreebank
import torch
from gensim.utils import tokenize

import model
from dataloader import SST
from glove_reader import GloveReader
from util import load_data


def validate(model, dev_dataset):
    dev = torch.utils.data.DataLoader(dev_dataset, batch_size=len(dev_dataset), num_workers=4, shuffle=False)
    
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

    model.train()
    return num_correct / total_count, tot_loss


def train(model, train_dataset, dev_dataset, max_epochs=100, model_name='model.save', stopping_counter=20):

    losses = []
    accs = []
    dev_losses = []
    dev_accs = []
    

    optimizer = torch.optim.Adam(model.parameters())

    loss_fn = torch.nn.CrossEntropyLoss()
    
    best_loss = float('+inf')
    best_model = model
    
    counter = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train = torch.utils.data.DataLoader(train_dataset, batch_size=256, num_workers=2, shuffle=True)
    model.train()
    for epoch in range(max_epochs):
        print("-" * 10 , "EPOCH ", epoch,  "-"*10)
        
        num_correct = 0.0
        total_count = 0.0
        start = time.time()
        epoch_loss = 0.0
        
        for i, batch in enumerate(train):
            if i + 1 % 100 == 0:
                print('Batch ', i)
            
            data, labels, mask = batch
            
            data = data.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            # print(data.device, mask.device, labels.device)

            optimizer.zero_grad()

            output = model(data, mask)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(output, 1)
            num_correct += (pred == labels).sum().item()
            total_count += pred.size(0)
            epoch_loss += loss.item()
        
        
        losses.append(epoch_loss / (i+1))
        accs.append(num_correct / total_count)
    
        end = time.time()
        
        counter += 1 # for early stopping
        
        eval_acc, eval_loss = validate(model, dev_dataset)
        
        print(f'Loss={losses[-1]}, Accuracy={accs[-1]}, Dev Accuracy={eval_acc}, epoch took {end - start}s')
        
        dev_losses.append(eval_loss)
        dev_accs.append(eval_acc)
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_model = model
            counter = 0
            print("Saving new best model...")
            torch.save(best_model.state_dict(), model_name)
        
        
        if counter == stopping_counter:
            return losses, accs, dev_losses, dev_accs
    
    return losses, accs, dev_losses, dev_accs

def main():
    #loads glove embeddings
    glove = GloveReader()

    #load train dataset
    train_data = SST(*load_data('./trees/train.txt', glove))


    # load dev dataset
    dev_data = SST(*load_data('./trees/dev.txt', glove))



    from itertools import product
    configuration = {
        'dropout': [0, 0.2],
        'hidden_size' : [256, 512],
        'n_layers': [1, 3],
        'embeddings': [glove.embeddings, None]
    }

    model_params = list(product(*configuration.values()))
    param_names = list(configuration)


    import pickle
    for params in model_params:
        info = f'{params[0]}_{params[1]}_{params[2]}_{"glove" if params[3] is not None else "default"}'
        model_name = f'model_{info}'
        data_name = f'data_{info}'

        m = model.RNN(300, params[2], params[1], 5, pretrained_embeddings=params[3], dropout=params[0])
        
        data = train(m, train_data, dev_data, model_name=model_name)
        
        with open(data_name, 'wb') as f:
            pickle.dump(data, f)
        
        del m
        del data

if __name__ == '__main__':
    main()
