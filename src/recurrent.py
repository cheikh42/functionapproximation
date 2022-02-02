from py_compile import main
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant


class RNN_f3(nn.Module):
    """Using 1 RNN layer and 1 linear layer"""

    def __init__(self, input_size=1, h=100, l=1):
        super(RNN_f3, self).__init__()

        # Number of hidden neurons
        self.hidden_dim = h

        # Number of hidden layers
        self.layer_dim = l

        # RNN
        self.rnn = nn.RNN(input_size, h, l, nonlinearity="relu")

        # Readout layer
        self.layer2 = nn.Linear(h, 25)

    def forward(self, x):

        h0 = Variable(torch.zeros(self.layer_dim, x.size(1), self.hidden_dim))
        # One time ste
        out, hn = self.rnn(x, h0)
        out = self.layer2(out[:, -1, :])
        return out
