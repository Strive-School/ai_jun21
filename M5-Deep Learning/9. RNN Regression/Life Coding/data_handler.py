import pandas as pd
import numpy as np
import torch


def get_clean_data(pth):


    df = pd.read_csv(pth)

    df['year'] = df['dteday'].apply(lambda y: int(y.split('-')[0]) )
    df['month'] = df['dteday'].apply(lambda y: int(y.split('-')[1]) )
    df['day'] = df['dteday'].apply(lambda y: int(y.split('-')[2]) )

    x = df.drop(['dteday', "instant", "casual", "registered"], axis = 1).values
    y = df['cnt'].values

    return x, y


x, y = get_clean_data("bikes.csv")

print(x.shape, y.shape)

def get_seq(x, y, seq_len = 7):

    x_data = []
    y_data = []

    for i in range(x.shape[0] - (seq_len + 1) ):

        x_data.append( x[i: i+seq_len,:]  )
        y_data.append(y[i+1: i+seq_len+1])

    return np.array(x_data), np.array(y_data)

x, y =  get_seq(x, y)

print(x.shape, y.shape)

def batchify(x, y, batch_size = 8):

    # randomize
    r = torch.randperm(x.shape[0])

    x = x[r]
    y = y[r]

    # cut extra
    n_batches = x.shape[0] // batch_size

    x , y = x[: n_batches * batch_size], y[: n_batches * batch_size]

    # put batches

    x = x.reshape(n_batches, batch_size, x.shape[1], x.shape[2])
    y = y.reshape(n_batches, batch_size, y.shape[1])

    return x, y



x, y = batchify(x, y)

print(x.shape, y.shape)