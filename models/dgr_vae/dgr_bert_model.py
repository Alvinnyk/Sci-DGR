import numpy as np
import random
import pandas as pd
from sklearn.metrics import f1_score, log_loss, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import seaborn as sns
import math

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler, RandomSampler, TensorDataset, SequentialSampler


class BertLSTM(nn.Module):

    def __init__(self, bert):
        super(BertLSTM, self).__init__()

        self.bidirectional = True
        self.rnn_num_directions = 2
        self.rnn_num_layers = 2
        self.rnn_hidden_size = 512

        self.bert = bert
        for param in self.bert.parameters():
            param.requires_grad = False

        self.rnn = nn.LSTM(input_size=768,
                           hidden_size=self.rnn_hidden_size,
                           bidirectional=self.bidirectional,
                           dropout=0.2,
                           num_layers=self.rnn_num_layers,
                           batch_first=True)

        self.fc1 = nn.Linear(512 * 2, 256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 3)
        self.dropout3 = nn.Dropout(p=0.5)
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask, hidden):
        batch_size, seq_len = sent_id.shape

        # pass the inputs to the model
        bert_outputs = self.bert(sent_id, attention_mask=mask)
        bert_embeddings = bert_outputs.last_hidden_state

        rnn_outputs, hidden = self.rnn(bert_embeddings, hidden)

        last_hidden = hidden[0].view(self.rnn_num_layers,
                                     self.rnn_num_directions,
                                     batch_size,
                                     self.rnn_hidden_size)[-1]

        h_1, h_2 = last_hidden[0], last_hidden[1]
        final_hidden = torch.cat((h_1, h_2), 1)  # Concatenate both states

        x = final_hidden

        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.softmax(x)

        return x

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.rnn_num_layers * self.rnn_num_directions,
                        batch_size,
                        self.rnn_hidden_size),
            torch.zeros(self.rnn_num_layers * self.rnn_num_directions,
                        batch_size,
                        self.rnn_hidden_size)
        )


def train_epoch_bert(model, loader, optimizer, criterion, device):
    epoch_loss = 0.0

    with tqdm(total=len(loader)) as progress_bar:
        for sent_id, mask, labels in loader:
            batch_size, seq_len = sent_id.shape[0], sent_id.shape[1]

            sent_id, mask, labels = sent_id.to(device), mask.to(device), labels.to(device)

            hidden = model.init_hidden(batch_size)
            hidden = (hidden[0].to(device), hidden[1].to(device))

            log_probs = model(sent_id, mask, hidden)

            loss = criterion(log_probs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.update(1)

    return epoch_loss


def train_bert(model, loader_train, loader_test, optimizer, criterion, num_epochs, device, verbose=False):
    results = []
    best_score = 0

    print("Total Training Time (total number of epochs: {})".format(num_epochs))
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = train_epoch_bert(model, loader_train, optimizer, criterion, device)

        model.eval()
        f1_train = evaluate_bert(model, loader_train, device)
        f1_test = evaluate_bert(model, loader_test, device)

        results.append((epoch_loss, f1_train, f1_test))

        if f1_test > best_score:
            model_checkpoint_path = 'data/model_bert.pth'
            torch.save(model, model_checkpoint_path)
            print("saving model checkpoint to {}".format(model_checkpoint_path))

            best_score = f1_test

        if verbose is True:
            print("[Epoch {}] loss:\t{:.3f}, f1 train: {:.3f}, f1 test: {:.3f} ".format(
                epoch, epoch_loss, f1_train, f1_test))

    return results


def evaluate_bert(model, loader, device, is_f1=True):
    y_true, y_pred = [], []

    with tqdm(total=len(loader)) as progress_bar:

        for sent_id, mask, labels in loader:
            batch_size, seq_len = sent_id.shape[0], sent_id.shape[1]

            # Move the batch to the correct device
            sent_id, mask, labels = sent_id.to(device), mask.to(device), labels.to(device)

            # Initialize the first hidden state h0 (and move to device)
            hidden = model.init_hidden(batch_size)

            hidden = (hidden[0].to(device), hidden[1].to(device))

            # Use model to compute log probabilities for each class
            log_probs = model(sent_id, mask, hidden)

            # Pick class with the highest log probability
            y_batch_pred = torch.argmax(log_probs, dim=1)

            y_true += list(labels.cpu().numpy())
            y_pred += list(y_batch_pred.cpu().numpy())

            # Update progress bar
            progress_bar.update(1)

    if is_f1:
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1
    else:
        return y_true, y_pred
