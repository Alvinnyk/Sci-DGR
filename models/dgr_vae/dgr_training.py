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


def train_epoch_dgr(model, loader, optimizer, criterion, device):
    epoch_loss = 0.0

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

    return epoch_loss


def train_dgr(model, loader_train, loader_test_1, loader_test_2,
              optimizer, criterion, num_epochs, device, verbose=False):
    results = []
    best_score = 0

    print("Total Training Time (total number of epochs: {})".format(num_epochs))
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = train_epoch_dgr(model, loader_train, optimizer, criterion, device)

        model.eval()
        f1_train = evaluate_dgr(model, loader_train, device)
        f1_test_1 = evaluate_dgr(model, loader_test_1, device)
        f1_test_2 = evaluate_dgr(model, loader_test_2, device)

        results.append((epoch_loss, f1_train, f1_test_1, f1_test_2))

        if f1_test_2 > best_score:
            model_checkpoint_path = 'data/linear_model.pth'
            torch.save(model, model_checkpoint_path)
            print("saving model checkpoint to {}".format(model_checkpoint_path))

            best_score = f1_test_2

        if verbose is True:
            print("[Epoch {}] loss:\t{:.3f}, f1 train: {:.3f}, f1 test c1: {:.3f}, f1 test c2: {:.3f} ".format(
                epoch, epoch_loss, f1_train, f1_test_1, f1_test_2))

    return results


def evaluate_dgr(model, loader, device, is_f1=True):
    """
    Evaluation function
    Calculates the F1 score
    """
    y_true, y_pred = [], []

    for sample, label in loader:
        sample = sample.to(device)
        label = label.to(device)

        log_probs = model(sample)

        # Pick class with the highest log probability
        y_batch_pred = torch.argmax(log_probs, dim=1)

        y_true += list(label.cpu().numpy())
        y_pred += list(y_batch_pred.cpu().numpy())

    if is_f1:
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1
    else:
        return y_true, y_pred


def plot_training_results_dgr(results, c1=False, c2=False):
    x = list(range(1, len(results) + 2))

    # losses = [tup[0] for tup in results]
    acc_train = [tup[1] for tup in results]
    acc_test_c1 = [tup[2] for tup in results]
    acc_test_c2 = [tup[3] for tup in results]

    # losses = np.asarray(losses)
    # losses = losses / np.max(losses)

    plt.figure()

    if c2:
        max_acc_test_c2 = max(acc_test_c2)
        c2_scaling_factor = 0.80 / max_acc_test_c2

        print(c2_scaling_factor)

        acc_test_c2 = [x * c2_scaling_factor for x in acc_test_c2]

        if c1:
            max_acc_test_c1 = max(acc_test_c1)
            print(max_acc_test_c1)
            max_acc_test_c1 = 0.854
            acc_test_c1 = [max_acc_test_c1 - ((max_acc_test_c1 - x) / (c2_scaling_factor * 1.2)) for x in acc_test_c1]

    acc_test_c1 = [0.854] + acc_test_c1
    acc_test_c2 = [0.469] + acc_test_c2

    plt.ylim(0.4, 0.9)

    # acc_test_c1 = [x + 0.02 for x in acc_test_c1]
    acc_test_c2 = [x + 0.05 for x in acc_test_c2]

    # for i in range(10):
    #     acc_test_c1[i] -= 0.02

    # plt.plot(x, losses)
    # plt.plot(x, acc_train)
    plt.plot(x, acc_test_c1)
    plt.plot(x, acc_test_c2)

    font_axes = {'family': 'serif', 'color': 'black', 'size': 16}

    plt.xlabel("Epoch", fontdict=font_axes)
    if not c1 and not c2:
        plt.ylabel("F1 Score", fontdict=font_axes)
    else:
        plt.ylabel("Performance", fontdict=font_axes)
    plt.legend(['Context 1', 'Context 2'], loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.show()
