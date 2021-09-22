from sklearn import model_selection
from sklearn import preprocessing
import pandas as pd
import numpy as np
import torch

def generate_dataset(pth, batch_size, shuffle=True):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = pd.read_csv(pth)
    dataset.drop(["instant"], axis=1, inplace = True)

    dataset["year"] = dataset.apply(lambda row: row["dteday"].split("-")[0], axis=1)
    dataset["month"] = dataset.apply(lambda row: row["dteday"].split("-")[1], axis=1)
    dataset["day"] = dataset.apply(lambda row: row["dteday"].split("-")[2], axis=1)
    dataset.drop(["dteday"], axis=1, inplace = True)

    y = dataset.values[:, -1]
    x = dataset.values[:, :-1]

    x, y = to_batches(x, y, batch_size, shuffle)

    x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y,
                                        test_size=0.2,
                                        random_state=0  # Recommended for reproducibility
                                    )

    transformer = preprocessing.PowerTransformer()
    y_train = transformer.fit_transform(y_train.reshape(-1,1))
    y_val = transformer.transform(y_val.reshape(-1,1))
    
    y_train = y_train.reshape(y_train.shape[0]//batch_size,batch_size,1)
    y_val = y_val.reshape(y_val.shape[0]//batch_size,batch_size,1)

    x_train = torch.tensor(x_train.astype(np.float32)).to(device)
    y_train = torch.tensor(y_train.astype(np.float32)).to(device)

    x_val = torch.tensor(x_val.astype(np.float32)).to(device)
    y_val = torch.tensor(y_val.astype(np.float32)).to(device)
    
    return x_train, x_val, y_train, y_val

def to_batches(x, y, batch_size, shuffle=True):

    if shuffle:
        indices = np.random.permutation(len(x))
        x = x[indices]
        y = y[indices]

        n_batches = len(x) // batch_size

        x = x[:n_batches * batch_size].reshape(n_batches, batch_size, x.shape[1])
        y = y[:n_batches * batch_size].reshape(n_batches, batch_size, 1)

    return x, y


#generate_dataset("bikes.csv", 10, shuffle=True)