import pickle

import torch
import numpy as np


def create_data():
    train_data = pickle.load(open("data/20ng/matlab/train.data.pkl", "rb"))
    test_data = pickle.load(open("data/20ng/matlab/test.data.pkl", "rb"))

    X = train_data.counts.toarray()
    # add a bit of normal text distribution to all data entries (acts as a Bayesian estimate of frequencies, can safely take log or divide by sum)
    X = X + train_data.f0[np.newaxis,:]
    X = torch.tensor(X).float()
    Y = torch.tensor(np.zeros(X.shape[0])).float()   # ignore labels
    train = torch.utils.data.TensorDataset(X, Y)

    X = test_data.counts.toarray()
    # add a bit of normal text distribution to all data entries (acts as a Bayesian estimate of frequencies, can safely take log or divide by sum)
    X = X + train_data.f0[np.newaxis,:]
    X = torch.tensor(X).float()
    Y = torch.tensor(np.zeros(X.shape[0])).float()   # ignore labels
    test = torch.utils.data.TensorDataset(X, Y)

    return train, test

