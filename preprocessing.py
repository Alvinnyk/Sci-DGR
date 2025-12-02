import pandas as pd
import re
import random

from collections import Counter, OrderedDict
from tqdm import tqdm

import torch
import torchtext
import spacy

import joblib


def load_dataset_raw(file_path, sample_column=None, columns_to_keep=None, drop_na=True):
    """
    Loads the json file
    """
    df = pd.read_json(file_path, lines=True)

    # drop all other columns, by default we keep all columns
    if columns_to_keep is not None:
        df = df[columns_to_keep]

    # drop columns with NA values
    if drop_na:
        df = df.dropna()

    # check for duplicates in the sample column, by default we don't check for duplicates
    if sample_column is not None:
        df = df.drop_duplicates(subset=sample_column)

    return df


def normalize_sentence(s):
    """
    Normalize a string based on regex rules
    """
    s = s.lower()  # Lowercase whole sentence
    s = re.sub(r'\s+', ' ', s)  # Remove duplicate whitespaces
    s = re.sub(r'([.]){2,}', ' ', s)  # Remove ellipses ...
    s = re.sub(r'([\w.-]+)([,;])([\w.-]+)', '\\g<1>\\g<2> \\g<3>', s)  # Add missing whitespace after , and ;
    s = re.sub(r'(.+)\1{2,}', '\\g<1>\\g<1>', s)  # Reduce repeated sequences to 2
    s = re.sub(r'(n\'t)', ' not', s)  # Resolve contraction "-n't"
    s = re.sub(r'\s+', ' ', s)  # Remove duplicate whitespaces (again)
    s = s.strip()  # Remove trailing whitespaces
    return s


def tokenize_sentence(s, nlp):
    """
    Tokenizes a string using a SpaCy language model
    """
    doc = nlp(s)
    token_list = []
    for token in doc:
        token_list.append(token.text)
    return token_list


def lemmatize_sentence(s, nlp):
    """
    Lemmatizes a string using a SpaCy language model
    """
    doc = nlp(s)
    token_list = []
    for token in doc:
        token_list.append(token.lemma_)
    return token_list


def extract_citation_tokens(text, cite_loc=None):
    """
    Citations can be generalized as a string contained between round, square or curly brackets
    This function extracts that pattern and replaces it with a special <cite> token

    If cite_loc is provided, it will extract the citation from the indicated indices
    """

    cite_token = "<cite>"

    if cite_loc is None:
        # citation_pattern = re.compile(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]')
        citation_pattern = re.compile(r'\((.*?)\)|\[(.*?)\]|\{(.*?)\}')

        output_text = re.sub(citation_pattern, cite_token, text)

        return output_text

    else:
        (cite_start, cite_end) = cite_loc
        cite_start = int(cite_start)
        cite_end = int(cite_end)

        output_text = text[:cite_start] + cite_token + text[cite_end + 1:]

        return output_text


def process_sentence(s, nlp, normalize=True, lemmatize=True, special_tokens=True):
    """
    Processes a sentence into a list of tokens
    normalize -> determines if the sentence should be normalized
    lemmatize -> apply lemmatization, else use simple tokenization
    special_tokens -> extract special tokens
    """
    if special_tokens:
        s = extract_citation_tokens(s)

    if normalize:
        s = normalize_sentence(s)

    if lemmatize:
        s = lemmatize_sentence(s, nlp)
    else:
        s = tokenize_sentence(s, nlp)

    return s


def initialize_tokenizer():
    """
    We will be using spaCy tokenizer to perform tokenization tasks
    en_core_web_sm: Small English model trained on web text (default model for English).
    """
    nlp = spacy.load("en_core_web_sm")

    special_token = "<cite>"
    nlp.tokenizer.add_special_case(special_token, [{spacy.attrs.ORTH: special_token}])

    return nlp


def process_data(df, label_to_index, nlp, normalize=True, lemmatize=True, special_tokens=True):
    """
    Processes a dataframe into a list of samples, list of labels and token_counter
    samples -> list of processed samples (normalized, tokenized)
    lables -> list of processed labels (mapped from label name to index)
    token_counter -> counter of tokens from the tokenized sentences
    """

    df_size = len(df)

    samples = []
    labels = []
    token_counter = Counter()

    with tqdm(total=df_size) as progress_bar:
        for index, row in df.iterrows():

            label = label_to_index[row['label']]
            text = row["string"]

            processed_text = process_sentence(text, nlp,
                                              normalize=normalize,
                                              lemmatize=lemmatize,
                                              special_tokens=special_tokens)

            samples.append(processed_text)
            labels.append(label)

            for token in processed_text:
                token_counter[token] += 1

            progress_bar.update(1)

    return samples, labels, token_counter


def shuffle_data(samples, labels):
    """
    Shuffles the dataset and labels
    """
    combined_data = list(zip(samples, labels))
    random.shuffle(combined_data)
    samples, labels = zip(*combined_data)

    return samples, labels


def initialize_vocabulary(token_counter, vocab_size=10000):
    """
    Initializes the vocabulary based on the token_counter
    """

    token_counter_sorted = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
    token_ordered_dict = OrderedDict(token_counter_sorted[:vocab_size])

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    SPECIALS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    vocabulary = torchtext.vocab.vocab(token_ordered_dict, specials=SPECIALS)

    vocabulary.set_default_index(vocabulary[UNK_TOKEN])

    return vocabulary


def save_vocabulary(vocabulary, vocabulary_file_path):
    """
    Saves the vocabulary file
    """
    torch.save(vocabulary, vocabulary_file_path)


def save_dataset(samples, labels, vocabulary, file_path):
    """
    Convert list of tokens into a list of indices based on the vocabulary
    Saves the list of indices into a text file
    """
    output_file = open(file_path, "w")

    for idx, text in enumerate(samples):
        label = labels[idx]

        # Convert tokens to their respective indices
        text_indexed = vocabulary.lookup_indices(text)

        # Write vectorized text and label to file
        output_file.write('{},{}\n'.format(' '.join([str(idx) for idx in text_indexed]), label))

    output_file.flush()
    output_file.close()


train_dataset_file_path = "data/raw/train.jsonl"
test_dataset_file_path = "data/raw/test.jsonl"

vocabulary_file_path = 'data/processed/vocabulary.vocab'
processed_train_dataset_file_path = 'data/processed/processed-data-train.txt'
processed_test_dataset_file_path = 'data/processed/processed-data-test.txt'

label_to_index = {
    "background": 0,
    "method": 1,
    "result": 2
}

if __name__ == "__main__":
    # Load dataset
    sample_column_name = "string"
    columns_to_keep = ['string', 'label']
    df_train = load_dataset_raw(train_dataset_file_path, sample_column=sample_column_name, columns_to_keep=columns_to_keep)
    df_test = load_dataset_raw(test_dataset_file_path, sample_column=sample_column_name, columns_to_keep=columns_to_keep)

    # initialize spaCy tokenizer
    nlp = initialize_tokenizer()

    # processing the dataset
    print("Processing train dataset...")
    texts_train, labels_train, token_counter = process_data(df_train, label_to_index, nlp)

    print("Processing test dataset...")
    texts_test, labels_test, _ = process_data(df_test, label_to_index, nlp)

    print('Total number of training samples: {}'.format(len(texts_train)))
    print('Total number of test samples: {}'.format(len(texts_test)))
    print('Number of unique tokens: {}'.format(len(token_counter)))

    # shuffle the dataset
    texts_train, labels_train = shuffle_data(texts_train, labels_train)

    # initialize vocabulary
    vocab = initialize_vocabulary(token_counter, vocab_size=10000)

    # save vocabulary file
    save_vocabulary(vocab, vocabulary_file_path)
    print('Vocabulary saved to {}'.format(vocabulary_file_path))

    # save the process training and testing dataset
    save_dataset(texts_train, labels_train, vocab, processed_train_dataset_file_path)
    print('Train Dataset saved to {}'.format(processed_train_dataset_file_path))
    save_dataset(texts_test, labels_test, vocab, processed_test_dataset_file_path)
    print('Test Dataset saved to {}'.format(processed_test_dataset_file_path))
