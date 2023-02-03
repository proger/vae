import sys
import tarfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import TensorDataset


def read_word_counts(file) -> torch.Tensor:
    # The .data files are formatted "docIdx wordIdx count" (1-indexed)
    data = torch.LongTensor([[int(i) for i in line.split()] for line in file])
    return torch.sparse_coo_tensor(data[:,:2].T - 1, data[:,2]).to_dense()


def unpack():
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


def make_datasets(vocab_size: int = 10000, frequency_smoothing: bool = True):
    train_counts, test_counts = unpack()
    
    vocabulary = torch.topk((train_counts>0).sum(dim=0), vocab_size).indices
    train_counts = train_counts[:, vocabulary].float()
    test_counts = test_counts[:, vocabulary].float()

    if frequency_smoothing:
        word_counts = train_counts.sum(dim=0).float()
        word_counts += 1./vocab_size
        frequencies = word_counts / word_counts.sum()

        train_counts += frequencies[None,:]
        test_counts += frequencies[None,:]

    return TensorDataset(train_counts), TensorDataset(test_counts)


if __name__ == '__main__':
    make_datasets()