import sys
from itertools import groupby
import pandas as pd
import numpy as np

window_size = 15
max_seq_len = 100

factorDict = {
    "A": [-0.59, -1.30, -0.73, 1.57, -0.15],
    "C": [-1.34, 0.47, -0.86, -1.02, -0.26],
    "D": [1.05, 0.30, -3.66, -0.26, -3.24],
    "E": [1.36, -1.45, 1.48, 0.11, -0.84],
    "F": [-1.01, -0.59, 1.89, -0.40, 0.41],
    "G": [-0.38, 1.65, 1.33, 1.05, 2.06],
    "H": [0.34, -0.42, -1.67, -1.47, -0.08],
    "I": [-1.24, -0.55, 2.13, 0.39, 0.82],
    "K": [1.83, -0.56, 0.53, -0.28, 1.65],
    "L": [-1.02, -0.99, -1.51, 1.27, -0.91],
    "M": [-0.66, -1.52, 2.22, -1.01, 1.21],
    "N": [0.95, 0.83, 1.30, -0.17, 0.93],
    "P": [0.19, 2.08, -1.63, 0.42, -1.39],
    "Q": [0.93, -0.18, -3.01, -0.50, -1.85],
    "R": [1.54, -0.06, 1.50, 0.44, 2.90],
    "S": [-0.23, 1.40, -4.76, 0.67, -2.65],
    "T": [-0.03, 0.33, 2.21, 0.91, 1.31],
    "V": [-1.34, -0.28, -0.54, 1.24, -1.26],
    "W": [-0.60, 0.01, 0.67, -2.13, -0.18],
    "Y": [0.26, 0.83, 3.10, -0.84, 1.51],
    "B": [1, 0.565, -1.18, -0.215, -1.155],
    "Z": [1.145, -0.815, -0.765, -0.195, -1.345],
    "J": [-1.13, -0.77, 0.31, 0.83, -0.045],
    "U": [-0.13, -0.12, 0.6, -0.03, -0.115],
    "O": [-0.13, -0.12, 0.6, -0.03, -0.115],
    "X": [-0.13, -0.12, 0.6, -0.03, -0.115]
}


def fasta_iter(fasta_name):
    with open(fasta_name) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in faiter:
            # Entire line, add .split[0] for just first column
            headerStr = header.__next__()[1:].strip().split()[0]

            seq = "".join(s.strip() for s in faiter.__next__())
            yield (headerStr, seq)


# function that converts fasta file to np


def source2pd(
        dataFile,
        window,
        maxLen
):

    print("IN:", dataFile)
    df = pd.read_csv(dataFile)

    def validate_seq(row, maxLen, window):
        z = len(row['sequences'])
        if window <= z <= maxLen:
            return row['sequences']
        else:
            return None

    df['sequences'] = df.apply(validate_seq, axis=1, args=(maxLen, window,))
    df = df.dropna(subset=['sequences'])
    return df


# --------------------
# Breaking up the Sequences into lengths of 15
# Artificially increases the number training and test sample !
# --------------------
def seq15mer(protein_panda, window, maxLen):
    # ,max_value=1000): n_samples=1000
    data = []

    for index, row in protein_panda.iterrows():
        headerStr, seq = row['headers'], row['sequences']
        seq = seq.strip("*")

        if len(seq) <= maxLen:
            for i in range(len(seq)):
                if len(seq) - window >= i:
                    data.append(
                        [headerStr, "kmer_" + str(i), seq[i:i + window]]
                    )
    data = np.array(data)

    return data


# ======================
# Input
# Positive samples file
# Negative samples file
# ======================
def seqs2train(
        pos_file,
        neg_file,
        window,
        maxLen
):
    pos_mat = []
    neg_mat = []

    # pos = ['./../sequence_data/training_data/pre.venom.fasta']
    # neg = ['./../sequence_data/training_data/pre.NOT.venom.fasta']
    # List of CSV files

    print(' > seqs2train ', pos_file, neg_file)


    pos_df = source2pd(pos_file, window, maxLen)
    neg_df = source2pd(neg_file, window, maxLen)

    # Split into train and test set
    # 80:20

    from sklearn.model_selection import train_test_split
    pos_train, pos_test = train_test_split (
        pos_df,
        test_size = 0.2,
        random_state = 42
    )

    neg_train, neg_test = train_test_split(
        neg_df,
        test_size=0.2,
        random_state=42
    )

    ...
    # if windowing is done
    if window:

        pos_train_data = seq15mer( pos_train, window,maxLen )
        pos_test_data = seq15mer( pos_test, window,maxLen )
        neg_train_data = seq15mer( neg_train, window,maxLen )
        neg_test_data = seq15mer( neg_test, window,maxLen )

    else:
        pos_train_data = pos_train.values
        pos_test_data = pos_test.values
        neg_test_data = neg_test.values
        neg_train_data = neg_train.values

    pos_np_train = pos_train_data
    pos_np_test  = pos_test_data
    neg_np_train = neg_train_data
    neg_np_test  = neg_test_data


    # --- Add in labels

    pos_ones_train  = np.ones((pos_np_train.shape[0], 1))
    pos_train_labeled = np.append(pos_np_train, pos_ones_train, axis=1)

    pos_ones_test = np.ones((pos_np_test.shape[0], 1))
    pos_test_labeled = np.append(pos_np_test, pos_ones_test, axis=1)

    neg_zeros_train = np.zeros((neg_np_train.shape[0], 1))
    neg_train_labeled = np.append(neg_np_train, neg_zeros_train, axis=1)

    neg_zeros_test = np.zeros((neg_np_test.shape[0], 1))
    neg_test_labeled = np.append(neg_np_test, neg_zeros_test, axis=1)

    data_train = np.vstack([pos_train_labeled, neg_train_labeled])
    data_test = np.vstack([pos_test_labeled, neg_test_labeled])


    print(' >>> Train data shape :', data_train.shape)
    print(' >>> Test data shape :', data_test.shape)
    return (data_train, data_test)


def seq2atchley(s, window, maxLen):
    seqList = []
    # print("WINDOW: ",window,window == True)
    if window:
        for i in range(len(s)):
            aa = s[i]
            seqList.append([])
            for factor in factorDict[aa]:
                seqList[i].append(factor)
    else:
        # print("SEQ:",s)
        for i in range(maxLen):

            try:
                # print("AMINO ACID:",i)
                aa = s[i]
                seqList.append([])
                for factor in factorDict[aa]:
                    seqList[i].append(factor)
            except:
                # print("amino acid: X")
                seqList.append([])
                for factor in factorDict["X"]:
                    seqList[i].append(0.0)
                    # NOTE, alternatively you could append factor as iff you were adding a bunch of Xs to the end

        # print("here will go zero-padding")
    # print(np.transpose(np.array(seqList)))
    return np.transpose(np.array(seqList))
