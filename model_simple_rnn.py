import numpy as np
import random
from sklearn.metrics import f1_score, log_loss, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler, RandomSampler, TensorDataset, SequentialSampler

vocabulary_file_path = 'data/processed/vocabulary.vocab'
processed_train_dataset_file_path = 'data/processed/processed-data-train.txt'
processed_test_dataset_file_path = 'data/processed/processed-data-test.txt'

MAX_SEQ_LEN = 256
BATCH_SIZE = 256

is_pretrained_embeddings = True
num_epochs = 10


class RnnTextClassifier(nn.Module):
    """
    Generalized RNN model for text classification
    """

    def __init__(self, params):
        super().__init__()

        # We have to memorize this for initializing the hidden state
        self.params = params

        # Calculate number of directions
        if self.params.rnn_bidirectional:
            self.rnn_num_directions = 2
        else:
            self.rnn_num_directions = 1

        #################################################################################
        ### Create layers
        #################################################################################

        # Embedding layer
        self.embedding = nn.Embedding(params.vocab_size, params.embed_size)

        # Recurrent Layer
        rnn_cell = self.params.rnn_cell
        if rnn_cell == "RNN":
            rnn = nn.RNN
        elif rnn_cell == "GRU":
            rnn = nn.GRU
        elif rnn_cell == "LSTM":
            rnn = nn.LSTM
        else:
            raise Exception("[Error] Unknown RNN Cell.")

        self.rnn = rnn(params.embed_size,
                       params.rnn_hidden_size,
                       num_layers=params.rnn_num_layers,
                       bidirectional=params.rnn_bidirectional,
                       dropout=params.rnn_dropout,
                       batch_first=True)

        # Linear layers (incl. Dropout and Activation)
        linear_sizes = ([params.rnn_hidden_size * self.rnn_num_directions]
                        + params.linear_hidden_sizes)

        self.linears = nn.ModuleList()
        for i in range(len(linear_sizes) - 1):
            self.linears.append(nn.Linear(linear_sizes[i], linear_sizes[i + 1]))
            self.linears.append(nn.ReLU())
            self.linears.append(nn.Dropout(p=params.linear_dropout))

        self.out = nn.Linear(linear_sizes[-1], params.output_size)

    #################################################################################

    def forward(self, inputs, hidden):

        batch_size, seq_len = inputs.shape

        # Push through embedding layer
        X = self.embedding(inputs)

        # Push through RNN layer
        rnn_outputs, hidden = self.rnn(X, hidden)

        # Extract last hidden state
        if self.params.rnn_cell == "LSTM":
            last_hidden = hidden[0].view(self.params.rnn_num_layers, self.rnn_num_directions, batch_size,
                                         self.params.rnn_hidden_size)[-1]
        else:
            last_hidden = \
                hidden.view(self.params.rnn_num_layers, self.rnn_num_directions, batch_size,
                            self.params.rnn_hidden_size)[
                    -1]

        # Handle directions
        if self.rnn_num_directions == 1:
            final_hidden = last_hidden.squeeze(0)
        elif self.rnn_num_directions == 2:
            h_1, h_2 = last_hidden[0], last_hidden[1]
            final_hidden = torch.cat((h_1, h_2), 1)  # Concatenate both states

        X = final_hidden

        # Push through linear layers (incl. Dropout & Activation layers)
        for l in self.linears:
            X = l(X)

        X = self.out(X)

        return F.log_softmax(X, dim=1)

    def init_hidden(self, batch_size):
        if self.params.rnn_cell == "LSTM":
            return (
                torch.zeros(self.params.rnn_num_layers * self.rnn_num_directions, batch_size,
                            self.params.rnn_hidden_size),
                torch.zeros(self.params.rnn_num_layers * self.rnn_num_directions, batch_size,
                            self.params.rnn_hidden_size))
        else:
            return torch.zeros(self.params.rnn_num_layers * self.rnn_num_directions, batch_size,
                               self.params.rnn_hidden_size)


class Dict2Class():
    """
    Converts a dictionary to class
    """

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


class BasicDataset(Dataset):
    """
    torch.utils.data.Dataset is an abstract class.
    provides an interface for accessing and retrieving individual samples from the dataset
    """

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.targets is None:
            return np.asarray(self.inputs[index])
        else:
            return np.asarray(self.inputs[index]), np.asarray(self.targets[index])


class EqualLengthsBatchSampler(Sampler):
    """
    Our inputs contain tensors of different lengths.
    This sampler will organize the tensors of equal length and output batches of tensors with equal length.
    """

    def __init__(self, batch_size, inputs, targets):

        # Throw an error if the number of inputs and targets don't match
        if targets is not None:
            if len(inputs) != len(targets):
                raise Exception("[EqualLengthsBatchSampler] inputs and targets have different sizes")

        # Remember batch size and number of samples
        self.batch_size, self.num_samples = batch_size, len(inputs)

        self.unique_length_pairs = set()
        self.lengths_to_samples = {}

        for i in range(0, len(inputs)):
            len_input = len(inputs[i])
            try:
                # Fails if targets[i] is not a sequence but a scalar (e.g., a class label)
                len_target = len(targets[i])
            except:
                # In case of failure, we just the length to 1 (value doesn't matter, it only needs to be a constant)
                len_target = 1

            # Add length pair to set of all seen pairs
            self.unique_length_pairs.add((len_input, len_target))

            # For each lengths pair, keep track of which sample indices for this pair
            # E.g.: self.lengths_to_sample = { (4,5): [3,5,11], (5,5): [1,2,9], ...}
            if (len_input, len_target) in self.lengths_to_samples:
                self.lengths_to_samples[(len_input, len_target)].append(i)
            else:
                self.lengths_to_samples[(len_input, len_target)] = [i]

        # Convert set of unique length pairs to a list so we can shuffle it later
        self.unique_length_pairs = list(self.unique_length_pairs)

    def __len__(self):
        return self.num_samples

    def __iter__(self):

        # Shuffle list of unique length pairs
        np.random.shuffle(self.unique_length_pairs)

        # Iterate over all possible sentence length pairs
        for length_pair in self.unique_length_pairs:

            # Get indices of all samples for the current length pairs
            # for example, all indices with a length pair of (8,7)
            sequence_indices = self.lengths_to_samples[length_pair]
            sequence_indices = np.array(sequence_indices)

            # Shuffle array of sequence indices
            np.random.shuffle(sequence_indices)

            # Compute the number of batches
            num_batches = np.ceil(len(sequence_indices) / self.batch_size)

            # Loop over all possible batches
            for batch_indices in np.array_split(sequence_indices, num_batches):
                yield np.asarray(batch_indices)


def load_vocabulary(file_path):
    """
    Loads vocabulary from file path
    """
    vocab = torch.load(file_path)
    return vocab


def load_dataset(file_path, max_length):
    """
    Loads dataset from file path
    Each sample will only have length of max_length tokens
    """
    samples = []
    with open(file_path) as file:
        for line in file:
            name, label = line.split(',')
            sequence = [int(index) for index in name.split()]
            samples.append((sequence[:max_length], int(label.strip())))

    return samples


def evaluate(model, loader, device, is_f1=True):
    """
    Evaluation function
    Calculates the F1 score
    """
    y_true, y_pred = [], []

    with tqdm(total=len(loader)) as progress_bar:

        for X_batch, y_batch in loader:
            batch_size, seq_len = X_batch.shape[0], X_batch.shape[1]

            # Move the batch to the correct device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Initialize the first hidden state h0 (and move to device)
            hidden = model.init_hidden(batch_size)

            if type(hidden) is tuple:
                hidden = (hidden[0].to(device), hidden[1].to(device))  # LSTM
            else:
                hidden = hidden.to(device)  # RNN, GRU

            # Use model to compute log probabilities for each class
            log_probs = model(X_batch, hidden)

            # Pick class with the highest log probability
            y_batch_pred = torch.argmax(log_probs, dim=1)

            y_true += list(y_batch.cpu().numpy())
            y_pred += list(y_batch_pred.cpu().numpy())

            # Update progress bar
            progress_bar.update(batch_size)

    # todo
    # refactor this function to always return y_true and y_pred instead
    # we will calculate metrics outside of this function
    if is_f1:
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1
    else:
        return y_true, y_pred



def train_epoch(model, loader, optimizer, criterion, device):
    """
    trains the model
    """
    epoch_loss = 0.0

    with tqdm(total=len(loader)) as progress_bar:

        for X_batch, y_batch in loader:
            batch_size, seq_len = X_batch.shape[0], X_batch.shape[1]

            # Move the batch to the correct device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Initialize the first hidden state h0 (and move to device)
            hidden = model.init_hidden(batch_size)

            if type(hidden) is tuple:
                hidden = (hidden[0].to(device), hidden[1].to(device))  # LSTM
            else:
                hidden = hidden.to(device)  # RNN, GRU

            log_probs = model(X_batch, hidden)

            # Calculate loss
            loss = criterion(log_probs, y_batch)

            optimizer.zero_grad()  # After each batch,set the gradients back to zero
            loss.backward()  # Calculating gradients using backpropagation
            optimizer.step()  # Update weights using the gradients

            # Keep track of overall epoch loss
            epoch_loss += loss.item()

            progress_bar.update(batch_size)

    return epoch_loss


def train(model, loader_train, loader_test, optimizer, criterion, num_epochs, device, verbose=False):
    """
    trains the model and saves it after each epoch
    """

    results = []

    print("Total Training Time (total number of epochs: {})".format(num_epochs))
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = train_epoch(model, loader_train, optimizer, criterion, device)

        model.eval()
        f1_train = evaluate(model, loader_train, device)
        f1_test = evaluate(model, loader_test, device)

        results.append((epoch_loss, f1_train, f1_test))

        model_checkpoint_path = f'model/rnn/rnn_model_epoch_{epoch}.pth'
        torch.save(model.state_dict(), model_checkpoint_path)
        print("saving model checkpoint to {}".format(model_checkpoint_path))

        if verbose is True:
            print("[Epoch {}] loss:\t{:.3f}, f1 train: {:.3f}, f1 test: {:.3f} ".format(
                epoch, epoch_loss, f1_train, f1_test))

    return results


if __name__ == '__main__':
    # Use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Available device: {}".format(device))
    if use_cuda:
        print(torch.cuda.get_device_name(0))

    # Load the preprocessed vocabulary and dataset
    vocabulary = load_vocabulary(vocabulary_file_path)
    vocab_size = len(vocabulary)
    samples_train = load_dataset(processed_train_dataset_file_path, MAX_SEQ_LEN)
    samples_test = load_dataset(processed_test_dataset_file_path, MAX_SEQ_LEN)

    # shuffle the dataset
    random.shuffle(samples_train)
    random.shuffle(samples_test)

    # convert the dataset into tensors
    X_train = [torch.LongTensor(seq) for (seq, _) in samples_train]
    X_test = [torch.LongTensor(seq) for (seq, _) in samples_test]

    y_train = [label for (_, label) in samples_train]
    y_test = [label for (_, label) in samples_test]

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Create data loaders
    dataset_train = BasicDataset(X_train, y_train)
    sampler_train = EqualLengthsBatchSampler(BATCH_SIZE, X_train, y_train)
    loader_train = DataLoader(dataset_train, batch_sampler=sampler_train, shuffle=False, drop_last=False)

    dataset_test = BasicDataset(X_test, y_test)
    sampler_test = EqualLengthsBatchSampler(BATCH_SIZE, X_test, y_test)
    loader_test = DataLoader(dataset_test, batch_sampler=sampler_test, shuffle=False, drop_last=False)

    # Model Parameters
    params = {
        "vocab_size": vocab_size,           # size of vocabulary
        "embed_size": 300,                  # 300 if using pretrained embeddings
        "rnn_cell": "GRU",                  # RNN, GRU or LSTM

        "rnn_num_layers": 2,                # number of rnn layers
        "rnn_bidirectional": True,          # go over each sequence from both directions

        "rnn_hidden_size": 512,             # size of the RNN hidden state
        "rnn_dropout": 0.5,                 # dropout for rnn layers

        "linear_hidden_sizes": [128, 64],   # list of sizes of hidden linear layers
        "linear_dropout": 0.5,              # dropout for linear layers

        "output_size": 3                    # number of output classes
    }
    params = Dict2Class(params)

    # Create model
    rnn = RnnTextClassifier(params).to(device)
    # Define optimizer
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.0001)

    # Define loss function
    class_weights = torch.tensor([1 / 4840, 1 / 2293, 1 / 1109]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.CrossEntropyLoss()

    # Print the model
    print(rnn)

    # Use pretrained word embeddings
    if is_pretrained_embeddings and params.embed_size == 300:
        print("Using Pretrained Embeddings")
        pretrained_vectors = torchtext.vocab.Vectors("data/embeddings/model.txt")
        pretrained_embedding = pretrained_vectors.get_vecs_by_tokens(vocabulary.get_itos())
        rnn.embedding.weight.data = pretrained_embedding

    # True -> embedding layer can be trained (not fixed)
    rnn.embedding.weight.requires_grad = False
    rnn.to(device)

    results = train(rnn, loader_train, loader_test, optimizer, criterion, num_epochs, device, verbose=True)
