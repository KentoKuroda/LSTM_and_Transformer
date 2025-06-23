import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.utils.rnn as rnn
import csv
import pandas as pd
import os
import random
import numpy as np
from datetime import datetime
import argparse

from lstm_kuroda import LSTMClassification

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from get_dataset import googledrive_download, init_dataset

# GPUチェック
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model')
    return parser.parse_args()


def main():

    args = parse_arguments()

    pretrained_model_path = args.pretrained_model

    # the number of players
    num_player = 22

    batch_size = 2048
    input_dim = (num_player + 1) * 2
    hidden_dim = 20
    target_size = 18
    num_epochs = 1000
    lr = 0.01

    # 各戦術的行動の名前 
    tactical_action_name_list = ['Build up 1', 'Progression 1', 'Final third 1', 'Counter-attack 1', 'High press 1', 'Mid block 1', 'Low block 1', 'Counter-press 1', 'Recovery 1', 'Build up 2', 'Progression 2', 'Final third 2', 'Counter-attack 2', 'High press 2', 'Mid block 2', 'Low block 2', 'Counter-press 2', 'Recovery 2']

    # numpy load
    sequence_np, label_np = googledrive_download() # _0_or_1, 
    print(sequence_np.shape, label_np.shape)

    label_np = label_np # [:, 1:]

    train_loader, val_loader, test_loader = init_dataset(sequence_np, label_np, batch_size)

    # モデルを定義
    model = LSTMClassification(input_dim=input_dim, hidden_dim=hidden_dim, target_size=target_size)

    history, model = train(train_loader, val_loader, test_loader, model, num_epochs, lr)
    torch.save(model.state_dict(), pretrained_model_path)


def train(train_loader, val_loader, test_loader, model, num_epochs, lr, mode='pretrain'):
    
    model = model.to(device)
    loss_function = nn.HuberLoss() # SmoothL1, CrossEntropyLoss, HuberLoss
    optimizer = optim.SGD(model.parameters(), lr=lr)

    history = {
        'loss': []
    }
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        val_losses = []

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()

            model.zero_grad()
            tag_scores = model(inputs)
            # labels = labels.unsqueeze(1)
            
            train_loss = loss_function(tag_scores, labels.float())
            
            train_loss.backward()
            optimizer.step()
            train_losses.append(float(train_loss))

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.long()
                model.zero_grad()
                tag_scores = model(inputs)
                val_loss = loss_function(tag_scores, labels.float())
                val_losses.append(float(val_loss))

        avg_train_loss = np.mean(train_losses)
        history['loss'].append(avg_train_loss)
        print("Epoch {} / {}: train_Loss = {:.3f}".format(epoch+1, num_epochs, avg_train_loss))

        avg_val_loss = np.mean(val_losses)
        history['loss'].append(avg_val_loss)
        print("Epoch {} / {}: val_Loss = {:.3f}".format(epoch+1, num_epochs, avg_val_loss))

    return history, model


if __name__ == "__main__":
    main()