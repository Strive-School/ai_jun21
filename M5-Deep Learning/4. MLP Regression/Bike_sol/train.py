import matplotlib.pyplot as plt
import data_handler as dh
from model import MLP
from torch import nn
import torch



def train(model, x, y, x_val, y_val, optimizer, criterion, epochs, device):

    loss_train = []
    loss_val = []
    best_loss = 1000000

    for epoch in range(epochs):
        losses = 0
        for i in range(x.shape[0]):

            model.to(device)

            optimizer.zero_grad()

            output = model.forward(x[i])

            #print(output.shape)
            #print(y[i].shape)

            loss = criterion(output, y[i])
            loss.backward()

            optimizer.step()

            losses += loss.item()

        if (epoch+1) % 2 == 0:
            loss_train.append(loss.item())
            test_losses = 0

            with torch.no_grad():

                for i in range(x_val.shape[0]):

                    output = model.forward(x_val[i])
                    loss = criterion(output, y_val[i])
                    test_losses += loss.item()
            
            if test_losses/x_val.shape[0] < best_loss:
                torch.save(model, 'model.pth')
                best_loss = test_losses/x_val.shape[0]

            loss_val.append(loss.item())
        
            print("Epoch: {}/{} \t Train Loss: {}, Test Loss: {}".format(epoch + 1, epochs, losses/x.shape[0], test_losses/x_val.shape[0]))
            print("\n")

    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.show()

x_train, x_val, y_train, y_val = dh.generate_dataset("bikes.csv", 10)


model = MLP(16, 8, 1)

opt = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.L1Loss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train(model, x_train, y_train, x_val, y_train, opt, criterion= criterion, epochs=200, device = device)