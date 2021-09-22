from torch import nn
from torchsummary import summary

class MLP(nn.Module):
    def __init__(self, input_size,  output_size, hidden_sizes = []):
        super(MLP, self).__init__()

        if len(hidden_sizes) == 0:
            self.layers = [ nn.Linear(input_size, output_size ) ]
        else:

            self.layers = [nn.Linear(input_size, hidden_sizes[0] )]

            for i in range(1,len(hidden_sizes)):
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            
            self.layers.append( nn.Linear(hidden_sizes[-1], output_size ))
            
            self.layers = nn.ModuleList(self.layers)

    def forward(self, x):

        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = nn.functional.relu(x)
        x = self.layers[-1](x)
        return x

        

model = MLP(8,1, hidden_sizes=[7,4])

print( summary(model,(8,)) )

