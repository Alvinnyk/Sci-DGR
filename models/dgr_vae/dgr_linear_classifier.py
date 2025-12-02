import numpy as np
import random
import pandas as pd
from sklearn.metrics import f1_score, log_loss, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import seaborn as sns

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler, RandomSampler, TensorDataset, SequentialSampler


class LinearClassifier(nn.Module):

    def __init__(self):
        super(LinearClassifier, self).__init__()

        self.fc1 = nn.Linear(512 * 2, 256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 3)
        self.dropout3 = nn.Dropout(p=0.5)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
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


def train_epoch_linear(model, loader, optimizer, criterion, device):
    epoch_loss = 0.0

    with tqdm(total=len(loader)) as progress_bar:
        for sample, label in loader:
            sample = sample.detach()
            sample = sample.to(device)

            label = label.to(device)

            log_probs = model(sample)

            loss = criterion(log_probs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.update(1)

    return epoch_loss


def train_linear(model, loader_train, loader_test, optimizer, criterion, num_epochs, device, verbose=False):
    results = []
    best_score = 0

    print("Total Training Time (total number of epochs: {})".format(num_epochs))
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = train_epoch_linear(model, loader_train, optimizer, criterion, device)

        model.eval()
        f1_train = evaluate_linear(model, loader_train, device)
        f1_test = evaluate_linear(model, loader_test, device)

        results.append((epoch_loss, f1_train, f1_test))

        if f1_test > best_score:
            model_checkpoint_path = 'data/linear_model.pth'
            torch.save(model, model_checkpoint_path)
            print("saving model checkpoint to {}".format(model_checkpoint_path))

            best_score = f1_test

        if verbose is True:
            print("[Epoch {}] loss:\t{:.3f}, f1 train: {:.3f}, f1 test: {:.3f} ".format(
                epoch, epoch_loss, f1_train, f1_test))

    return results


def evaluate_linear(model, loader, device, is_f1=True):
    """
    Evaluation function
    Calculates the F1 score
    """
    y_true, y_pred = [], []

    with tqdm(total=len(loader)) as progress_bar:

        for sample, label in loader:
            sample = sample.to(device)
            label = label.to(device)

            log_probs = model(sample)

            # Pick class with the highest log probability
            y_batch_pred = torch.argmax(log_probs, dim=1)

            y_true += list(label.cpu().numpy())
            y_pred += list(y_batch_pred.cpu().numpy())

            # Update progress bar
            progress_bar.update(1)

    # todo
    # refactor this function to always return y_true and y_pred instead
    # we will calculate metrics outside of this function
    if is_f1:
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1
    else:
        return y_true, y_pred
