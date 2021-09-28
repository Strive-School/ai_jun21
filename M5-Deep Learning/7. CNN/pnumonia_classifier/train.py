import torch
from torch import nn
from torch.optim import Adam
import data_handler as dh
from model import CNN_CLF
from torchsummary import summary

#TRAINING



model = CNN_CLF()

print( summary( model, (1,512,512) ) )

criterion = nn.CrossEntropyLoss()
lr = 0.001
optim = Adam(model.parameters(), lr )

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()


best_val = float('INF')

epochs = 5

