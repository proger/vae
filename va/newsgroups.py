import sys
import tarfile
import urllib.request
from pathlib import Path

import torch


def read_word_counts(file) -> torch.Tensor:
    # The .data files are formatted "docIdx wordIdx count" (1-indexed)
    data = torch.LongTensor([[int(i) for i in line.split()] for line in file])
    return torch.sparse_coo_tensor(data[:,:2].T - 1, data[:,2]).to_dense()


def unpack_data():
    url = 'http://qwone.com/~jason/20Newsgroups/20news-bydate-matlab.tgz'
    data = Path('data')
    data.mkdir(exist_ok=True, parents=True)
    filename = data / '20news-bydate-matlab.tgz'

    if not filename.exists():
        print('Downloading 20 Newsgroups dataset...', file=sys.stderr)
        urllib.request.urlretrieve(url, filename)

    with tarfile.open(filename) as tar:
        train_data = read_word_counts(tar.extractfile('20news-bydate/matlab/train.data'))
        test_data = read_word_counts(tar.extractfile('20news-bydate/matlab/test.data'))

    return train_data, test_data


def prepare_data():
    # load the word counts data provided from the original dataset:
    train_counts, test_counts = unpack_data()
    
    global_word_counts = (train_counts>0).sum(dim=0)
    vocabulary = torch.topk(global_word_counts, 10000).indices

    # keep only vocabulary words
    train_counts = train_counts[:, vocabulary]

    # word frequencies in the whole corpus
    counts = train_counts.sum(dim=0) # number of times words from the selected vocabulary appear in train set
    counts_smoothed = counts + 1/len(counts) # increase the count of each word by 1/10000 (regularization)
    frequencies = counts_smoothed / counts_smoothed.sum() # word frequencies in the whole training set

    test_counts = test_counts[:, vocabulary]

    return train_counts, frequencies, test_counts


def create_data():
    train_counts, frequencies, test_counts = prepare_data()

    # add a bit of normal text distribution to all data entries (acts as a Bayesian estimate of frequencies, can safely take log or divide by sum)
    train = torch.utils.data.TensorDataset(train_counts + frequencies[None,:])
    test = torch.utils.data.TensorDataset(test_counts + frequencies[None,:])

    return train, test


if __name__ == '__main__':
    create_data()