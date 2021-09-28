from torch import nn
from torchsummary import summary


class CNN_CLF(nn.Module):

    def __init__(self):
        super(CNN_CLF, self).__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size = 3, stride = 2, padding = 1),
            nn.Dropout(0.2, inplace = True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 32, kernel_size = 5, stride = 2, padding = 1),
            nn.Dropout(0.2, inplace = True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 64, kernel_size = 5, stride = 2, padding = 1),
            nn.Dropout(0.2, inplace = True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
        )

        self.linear = nn.Sequential(
            nn.Linear(64*63*63, 128),
            nn.Dropout(0.2, inplace = True),

            nn.Linear(128, 16),
            nn.Dropout(0.2, inplace = True),

            nn.Linear(16, 2),

        )

    def forward(self, x):

        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x  = self.linear(x)
        return x

