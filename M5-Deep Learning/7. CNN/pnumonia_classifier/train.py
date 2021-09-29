import torch
from torch import nn
from torch.optim import Adam
import data_handler as dh
from model import CNN_CLF
from torchsummary import summary
from matplotlib import pyplot as plt
from tqdm import tqdm

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

        x_train , y_train = dh.batchify(x_train , y_train)

        loss_iter = 0

        for x , y in tqdm(zip(x_train, y_train)):
            
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
epochs = 5
lr = 0.001

#Models
model = CNN_CLF().float()
#print( summary( model, (1,512,512) ) )

criterion = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr )

#Device
model = model.to(device)
criterion = criterion.to(device)

#TRAINING

train(epochs, x_train , y_train, x_test, y_test, model, criterion, optim)



