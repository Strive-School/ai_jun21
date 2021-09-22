from operator import index
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def load_data(pth, batch_size):

    data = pd.read_csv(pth)

    data.drop(['date'], axis = 1, inplace = True)

    y = data['DAX'].values
    x = data.drop(['DAX'], axis = 1).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = torch.tensor(x_train.astype(np.float32)).to(device)
    x_test = torch.tensor(x_test.astype(np.float32)).to(device)

    y_train = torch.tensor(y_train.astype(np.float32)).to(device)
    y_test = torch.tensor(y_test.astype(np.float32)).to(device)

    return x_train, x_test, y_train, y_test

#(100, 7) -> (10,10,7)
#(100, 1) -> (10,10,1)

def to_batches(x_train, x_test, y_train, y_test, batch_size):

    n_batches = x_train.shape[0] // batch_size # 11 / 3 = 3.66 -> 3
    n_batches_test = x_test.shape[0] // batch_size

    indexes = np.random.permutation(x_train.shape[0])
    indexes_test = np.random.permutation(x_test.shape[0])


    x_train = x_train[indexes]
    y_train = y_train[indexes]

    x_test = x_test[indexes_test]
    y_test = y_test[indexes_test]

    x_train = x_train[ :batch_size * n_batches ].reshape(n_batches, batch_size, x_train.shape[1])
    y_train = y_train[ :batch_size * n_batches ].reshape(n_batches, batch_size, 1)
    
    x_test = x_test[ :batch_size * n_batches_test ].reshape(n_batches_test, batch_size, x_test.shape[1])
    y_test = y_test[ :batch_size * n_batches_test ].reshape(n_batches_test, batch_size, 1)


    return x_train, x_test, y_train, y_test
