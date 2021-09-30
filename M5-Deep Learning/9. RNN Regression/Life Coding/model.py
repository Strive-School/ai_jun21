from torch import nn
import torch

class Model(nn.Module):

    def __init__(self, input_size, neurons, hidden_size, keep_memory = True):
        super(Model, self).__init__()

        self.neurons = neurons
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.keep_memory = keep_memory

        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.neurons, batch_first = True )

        if self.keep_memory:
            self.hidden = torch.zeros(self.input_size, self.hidden_size, self.neurons)

    def forward(self, x):
        
        if not self.keep_memory:
            self.hidden = torch.zeros(self.input_size, self.hidden_size, self.neurons)
        
        _, hid = self.rnn(x, self.hidden)

        hid = hid[-1]

        out = hid.flatten()

        return out
            
