import torch
import numpy as np
from PIL import Image
import os

torch.manual_seed(0)


def get_data(pth):
    
    x , y = [], []

    for folder in os.listdir(pth):
        for img_pth in os.listdir( os.path.join(pth, folder) ):

            img = Image.open( os.path.join(pth, folder,img_pth) ).convert("L").resize((512,512))
            img = np.array(img).astype(np.float64)

            img = torch.from_numpy(img).reshape(1,512,512)
            x.append(img)

            y.append( 1 if folder == "NORMAL" else 0 )

    x = torch.stack(x)

    y = np.array(y).astype(int)
    y = torch.from_numpy(y)

    y = y.unsqueeze(1)


    return x, y


def batchify(x, y, batch_size = 32):

    # randomize
    r = torch.randperm(x.shape[0])

    x = x[r]
    y = y[r]

    # cut extra
    n_batches = x.shape[0] // batch_size

    x , y = x[: n_batches * batch_size], y[: n_batches * batch_size]

    # put batches

    x = x.reshape(n_batches, batch_size, x.shape[1], x.shape[2], x.shape[2])
    y = y.reshape(n_batches, batch_size, y.shape[1])

    return x, y

x, y = get_data("chest_xray/train")

x, y = batchify(x, y)

print(x.shape, y.shape)
