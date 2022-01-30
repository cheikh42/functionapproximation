from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch import nn


class NN0_0(nn.Module):
    """Using 1 layer with 4 neurons"""

    def __init__(self, input_features=1, h1=4):
        super().__init__()
        self.layer1 = nn.Linear(input_features, h1)
        self.output = nn.Linear(h1, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.output(x)
        return x


class NN1_F_D_1_ReLu(nn.Module):
    """4 hidden layers, Each has 10 neurons
    Using the ReLu activation function for all the layer
    """

    def __init__(self, input_features=2):
        super().__init__()
        self.layer1 = nn.Linear(input_features, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.layer4 = nn.Linear(10, 10)
        self.layer5 = nn.Linear(10, 10)

        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = self.output(x)
        return x


class NN1_F_D_2_ReLu(nn.Module):
    """5 hidden layers, each has 10 neurons"""

    def __init__(self, input_features=2):
        super().__init__()
        self.layer1 = nn.Linear(input_features, 10)

        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.layer4 = nn.Linear(10, 10)
        self.layer5 = nn.Linear(10, 10)
        self.layer6 = nn.Linear(10, 10)

        self.output = nn.Linear(10, 1)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))

        x = self.output(x)

        return x


class NN1_F_D_3_ReLu(nn.Module):
    """
    8 hidden layers, each has 10 neurons
    """

    def __init__(self, input_features=2):
        super().__init__()
        self.layer1 = nn.Linear(input_features, 10)

        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.layer4 = nn.Linear(10, 10)
        self.layer5 = nn.Linear(10, 10)
        self.layer6 = nn.Linear(10, 10)
        self.layer7 = nn.Linear(10, 10)
        self.layer8 = nn.Linear(10, 10)

        self.output = nn.Linear(10, 1)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))

        x = self.output(x)

        return x


class NN1_F_D_4_ReLu(nn.Module):
    """
    10 hidden layers, each has 10 neurons
    """

    def __init__(self, input_features=2):
        super().__init__()
        self.layer1 = nn.Linear(input_features, 10)

        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.layer4 = nn.Linear(10, 10)
        self.layer5 = nn.Linear(10, 10)
        self.layer6 = nn.Linear(10, 10)
        self.layer7 = nn.Linear(10, 10)
        self.layer8 = nn.Linear(10, 10)
        self.layer9 = nn.Linear(10, 10)
        self.layer10 = nn.Linear(10, 10)
        self.layer11 = nn.Linear(10, 10)

        self.output = nn.Linear(10, 1)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        x = F.relu(self.layer9(x))
        x = F.relu(self.layer10(x))
        x = F.relu(self.layer11(x))

        x = self.output(x)

        return x


class NN1_F_S(nn.Module):
    """
    2 hidden layers each has 50 neurons
    """

    def __init__(self, input_features=2, h1=50, h2=50):
        super().__init__()
        self.layer1 = nn.Linear(input_features, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.output = nn.Linear(h2, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return x
