from py_compile import main
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch import nn
import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant


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

    def __init__(self, input_features=2, h=10):
        super().__init__()
        self.layer1 = nn.Linear(input_features, h)
        self.layer2 = nn.Linear(h, h)
        self.layer3 = nn.Linear(h, h)
        self.layer4 = nn.Linear(h, h)
        self.layer5 = nn.Linear(h, h)

        self.output = nn.Linear(h, 1)

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

    def __init__(self, input_features=2, h=10):
        super().__init__()
        self.layer1 = nn.Linear(input_features, h)

        self.layer2 = nn.Linear(h, h)
        self.layer3 = nn.Linear(h, h)
        self.layer4 = nn.Linear(h, h)
        self.layer5 = nn.Linear(h, h)
        self.layer6 = nn.Linear(h, h)

        self.output = nn.Linear(h, 1)

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

    def __init__(self, input_features=2, h=10):
        super().__init__()
        self.layer1 = nn.Linear(input_features, h)

        self.layer2 = nn.Linear(h, h)
        self.layer3 = nn.Linear(h, h)
        self.layer4 = nn.Linear(h, h)
        self.layer5 = nn.Linear(h, h)
        self.layer6 = nn.Linear(h, h)
        self.layer7 = nn.Linear(h, h)
        self.layer8 = nn.Linear(h, h)

        self.output = nn.Linear(h, 1)

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

    def __init__(self, input_features=2, h=10):
        super().__init__()
        self.layer1 = nn.Linear(input_features, h)

        self.layer2 = nn.Linear(h, h)
        self.layer3 = nn.Linear(h, h)
        self.layer4 = nn.Linear(h, h)
        self.layer5 = nn.Linear(h, h)
        self.layer6 = nn.Linear(h, h)
        self.layer7 = nn.Linear(h, h)
        self.layer8 = nn.Linear(h, h)
        self.layer9 = nn.Linear(h, h)
        self.layer10 = nn.Linear(h, h)
        self.layer11 = nn.Linear(h, h)

        self.output = nn.Linear(h, 1)

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


class NN0_F_S(nn.Module):
    """
    1 hidden layers each has 50 neurons
    """

    def __init__(self, input_features=2, h1=50):
        super().__init__()
        self.layer1 = nn.Linear(input_features, h1)
        self.output = nn.Linear(h1, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.output(x)
        return x


class NN1_F_S(nn.Module):
    """
    2 hidden layers each has 50 neurons by default
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
