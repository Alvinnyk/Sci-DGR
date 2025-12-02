import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_dataset_raw(file_path, sample_column=None, columns_to_keep=None, drop_na=True):
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


def process_data_bert(df, label_to_index):
    df_size = len(df)

    samples = []
    labels = []

    with tqdm(total=df_size) as progress_bar:
        for index, row in df.iterrows():
            label = label_to_index[row['label']]
            text = row["string"]

            samples.append(text)
            labels.append(label)

            progress_bar.update(1)

    return samples, labels


def plot_training_results(results):
    x = list(range(1, len(results) + 1))

    losses = [tup[0] for tup in results]
    acc_train = [tup[1] for tup in results]
    acc_test = [tup[2] for tup in results]

    losses = np.asarray(losses)
    losses = losses / np.max(losses)

    plt.figure()

    plt.plot(x, losses)
    plt.plot(x, acc_train)
    plt.plot(x, acc_test)

    font_axes = {'family': 'serif', 'color': 'black', 'size': 16}

    plt.xlabel("Epoch", fontdict=font_axes)
    plt.ylabel("F1 Score", fontdict=font_axes)
    plt.legend(['Loss', 'F1 (train)', 'F1 (test)'], loc='lower left', fontsize=12)
    plt.tight_layout()
    plt.show()
