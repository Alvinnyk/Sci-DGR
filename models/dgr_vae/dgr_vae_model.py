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


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # Outputting mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode input
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode latent variable
        decoded = self.decoder(z)

        return decoded, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # loss = F.mse_loss(recon_x, x.view(-1, 1024), reduction='sum')
    # loss = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 1024), reduction='sum')
    loss = F.smooth_l1_loss(recon_x, x.view(-1, 1024), reduction='sum')
    # loss = F.smooth_l1_loss(recon_x, x.view(-1, 1024), reduction='mean')

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = loss + kld

    return total_loss


def train_epoch_vae(vae, loader, optimizer, device):
    epoch_loss = 0.0

    vae.train()

    for x, y in loader:
        data = x.to(device)

        recon_batch, mu, logvar = vae(data)

        loss = loss_function(recon_batch, data, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def eval_vae(vae, loader, device):
    val_loss = 0.0
    vae.eval()

    for x, y in loader:
        data = x.to(device)

        recon_batch, mu, logvar = vae(data)

        loss = loss_function(recon_batch, data, mu, logvar)

        val_loss += loss.item()

    return val_loss / len(loader)


class FeatureExtractor(nn.Module):

    def __init__(self, model):
        super(FeatureExtractor, self).__init__()

        self.bert = model.bert

        self.bidirectional = model.bidirectional
        self.rnn_num_directions = model.rnn_num_directions
        self.rnn_num_layers = model.rnn_num_layers
        self.rnn_hidden_size = model.rnn_hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False

        self.rnn = model.rnn

        for param in self.rnn.parameters():
            param.requires_grad = False

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


class FeatureClassifier(nn.Module):

    def __init__(self, model):
        super(FeatureClassifier, self).__init__()

        self.fc1 = model.fc1
        self.dropout1 = model.dropout1
        self.relu1 = model.relu1

        self.fc2 = model.fc2
        self.dropout2 = model.dropout2
        self.relu2 = model.relu2

        self.fc3 = model.fc3
        self.dropout3 = model.dropout3
        self.softmax = model.softmax

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
