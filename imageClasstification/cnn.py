import torch.nn as nn
import torch
import torch.nn.functional as F


class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Input: 3x32x32, 12 filters (5x5), stride=1, no padding -> Output: 12x28x28.
        self.conv1 = nn.Conv2d(3, 12, 5) 
        self.pool = nn.MaxPool2d(2, 2) #first pooling: (12, (28/2), (28/2)) = (12, 14, 14) 2nd pooling: (24, (10/2), (10/2)) = (24, 5, 5)
        #Input: 12x14x14, 24 filters (5x5), stride=1, no padding -> Output: 24x10x10.
        self.conv2 = nn.Conv2d(12, 24, 5)
        # Flatten: 24x5x5 feature maps -> 600 features.
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p= 0.3)
    

    def forward(self, x):
        #We feed the input into the first convolution and then feed the convolution to the activation function and then pool it
        # Conv1 (3* 32 * 32 -> 12*28*28), ReLU, pool (12*28*28) -> 12*14*14)
        x = self.pool(F.relu(self.conv1(x))) 
        #We feed the input into the 2nd convolution and then feed the convolution to the activation function and then pool it
        # Conv2 (12x14x14 -> 24x10x10), ReLU, pool (24x10x10 -> 24x5x5)
        x = self.pool(F.relu(self.conv2(x)))
        # flatten 24x5x5 feature maps to 600-element vector.
        x = torch.flatten(x, 1)
        #After flattening it, we then feed it into the first fully connected layer and then feed it into the relu activation function
        # fc1: 600 -> 120 neurons, then ReLU.
        x = self.dropout(F.relu(self.fc1(x)))
        # fc2: 120 -> 84 neurons, then ReLU
        x =  self.dropout(F.relu(self.fc2(x)))
        # fc3: 84 -> 5 logits (no activation, for 5-class classification).
        x = self.fc3(x)

        return x
