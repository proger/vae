import torch


def prepare_data():
    # load the word counts data provided from the original dataset:
    
    with open("data/20ng/matlab/train.data") as f:
        train_data = torch.LongTensor([[int(i) for i in line.split()] for line in f])
        train_counts = torch.sparse_coo_tensor(train_data[:,:2].T - 1, train_data[:,2]).to_dense()

    global_word_counts = (train_counts>0).sum(dim=0)
    vocabulary = torch.topk(global_word_counts, 10000).indices

    # keep only vocabulary words
    train_counts = train_counts[:, vocabulary]

    # word frequencies in the whole corpus
    counts = train_counts.sum(dim=0) # number of times words from the selected vocabulary appear in train set
    counts_smoothed = counts + 1/len(counts) # increase the count of each word by 1/10000 (regularization)
    frequencies = counts_smoothed / counts_smoothed.sum() # word frequencies in the whole training set

    # load the test set
    with open("data/20ng/matlab/test.data") as f:
        test_data = torch.LongTensor([[int(i) for i in line.split()] for line in f])
        test_counts = torch.sparse_coo_tensor(test_data[:,:2].T - 1, test_data[:,2]).to_dense()

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