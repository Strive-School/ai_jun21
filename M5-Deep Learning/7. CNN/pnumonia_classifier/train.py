#utils
from matplotlib import pyplot as plt
from torchsummary import summary
from tqdm import tqdm
import argparse

#torch
from torch.optim import Adam
from torch import nn
import torch
#files
import data_handler as dh
from model import CNN_CLF


parser = argparse.ArgumentParser(description="PNEUMONIA CNN")
parser.add_argument('--epochs', metavar = 'e', type = int, required = True)
parser.add_argument('--lr', metavar = 'l', type = float, required = True)
parser.add_argument('--dropout', metavar = 'd', type = float, required = True)
args = vars(parser.parse_args())

def validate(x_test, y_test, model, criterion, device):

    with torch.no_grad():

        x_test, y_test = dh.batchify(x_test, y_test)

        val_loss = 0

        for x , y in zip(x_test, y_test):
            

            x, y = x.to(device), y.to(device)

            out = model(x)

            loss = criterion(out, y)

            val_loss+= loss.cpu().item()

        return val_loss / x_test.shape[0]





def train(epochs, x_train , y_train, x_test, y_test, model, criterion, optim):

    best_val = float('INF')
    model.train()
    train_losses = []
    val_losses = []


    for epoch in range( epochs ):

        x_train_batches , y_train_batches  = dh.batchify(x_train , y_train)

        loss_iter = 0

        for x , y in tqdm(zip(x_train_batches , y_train_batches )):
            
            optim.zero_grad()

            x, y = x.to(device), y.to(device)

            out = model(x)

            loss = criterion(out, y)
            loss.backward()

            optim.step()

            loss_iter += loss.cpu().item()

        if epoch+1 % 2:

            val_loss = validate(x_test, y_test, model, criterion, device)
            print("Epoch: {} \t Train Loss: {} \t Validation Loss: {}". format(epoch, loss_iter/x_train.shape[0], val_loss))

            if val_loss < best_val:
                print("Saving Model")
                best_val = val_loss
                torch.save(model, "models/best_val_model.pth")
                
                

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.savefig("figures/Losses.png")
    
#Data
x_train, y_train = dh.get_data("chest_xray/train")
print(x_train.shape)

x_test, y_test = dh.get_data("chest_xray/test")
print(x_test.shape)#Device
device = torch.device("cuda" if  torch.cuda.is_available() else "cpu" )

#Placeholders



#HyperParams
epochs = args['epochs']
lr = 0.001

#Models
model = CNN_CLF(args['dropout']).float()
#print( summary( model, (1,512,512) ) )

criterion = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), args['lr'] )

#Device
model = model.to(device)
criterion = criterion.to(device)

#TRAINING

train(epochs, x_train , y_train, x_test, y_test, model, criterion, optim)



