import pickle

import torch
import numpy as np

import scipy


def prepare_data():
    # load the word counts data provided from the original dataset:
    train_data = np.loadtxt("data/20ng/matlab/train.data", dtype='i', delimiter=' ')
    train_counts = scipy.sparse.csc_matrix((train_data[:,2], (train_data[:,0] - 1, train_data[:,1] - 1))) # 0-based indexing
    
    S0 = np.asarray((train_counts>0).sum(axis=0)).flatten() # number of times word used in the train data
    # select 10000 most frequent words and save data (from end-10000 till the end)
    ind = np.argsort(S0)[-10000:]        
    train_counts = train_counts[:,ind]

    # check any empty documents
    assert((train_counts.sum(axis=1) ==0).sum() == 0)

    # word frequencies in the whole corpus
    train_f0 = np.asarray(train_counts.sum(axis=0)).flatten() # number of times wors from the selected subvocabulary appear in train set
    train_f0 = train_f0 + 1/len(train_f0) # increase the count of each work by 1/10000 (regularization)
    train_f0 = train_f0 / train_f0.sum() # word frequencies in the whole training set

    # same for the test data
    test_data = np.loadtxt("data/20ng/matlab/test.data", dtype='i', delimiter=' ')
    test = scipy.sparse.csc_matrix((test_data[:,2], (test_data[:,0] - 1, test_data[:,1] - 1))) # 0-based indexing
    test_counts = test[:, ind]

    # word frequencies in the whole corpus
    test_f0 = np.asarray(test_counts.sum(axis=0)).flatten()
    test_f0 = test_f0 + 1/len(test_f0)
    test_f0 = test_f0 / test_f0.sum()

    return train_counts, train_f0, test_counts, test_f0


def load_data_old():
    train_data = pickle.load(open("data/20ng/matlab/train.data.pkl", "rb"))
    test_data = pickle.load(open("data/20ng/matlab/test.data.pkl", "rb"))
    
    return train_data.counts, train_data.f0, test_data.counts, test_data.f0


def create_data():
    train_counts, train_f0, test_counts, test_f0 = prepare_data()

    X = train_counts
    # add a bit of normal text distribution to all data entries (acts as a Bayesian estimate of frequencies, can safely take log or divide by sum)
    X = X + train_f0[np.newaxis,:]
    X = torch.tensor(X).float()
    Y = torch.tensor(np.zeros(X.shape[0])).float()   # ignore labels
    train = torch.utils.data.TensorDataset(X, Y)

    X = test_counts.toarray()
    # add a bit of normal text distribution to all data entries (acts as a Bayesian estimate of frequencies, can safely take log or divide by sum)
    X = X + train_f0[np.newaxis,:]
    X = torch.tensor(X).float()
    Y = torch.tensor(np.zeros(X.shape[0])).float()   # ignore labels
    test = torch.utils.data.TensorDataset(X, Y)

    return train, test


if __name__ == '__main__':
    create_data()