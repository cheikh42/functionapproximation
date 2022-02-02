from py_compile import main
import torch
from torch import nn
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm


def forward_model_training_0_1(epochs: int, i: int, lr: float, model: nn.Module, alpha):

    """
    Train the feed-forward neural_network model
    Args:
        epochs
        i: index of the function
        lr: learning rate
        model: The neural network model you want to train the func on
        alpha: The regularization function

    Returns:
        loss_func_relu (list): A list of each loss rate
        model (nn.Module): A list of each loss rate
    """
    # Creating an instance of the model
    model_1 = model
    # Getting the tensors
    Input_train, f_train, Input_test, f_test, f_data, Input_data = data_splitter(i)

    # specifiying criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adamax(model_1.parameters(), lr=lr, weight_decay=alpha)

    loss_func_relu = []

    for e in tqdm(range(epochs)):

        # forward pass
        output = model_1(Input_train)
        # compute the loss
        loss = criterion(output, f_train)
        # clear previous gradients
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        optimizer.step()

        loss_func_relu.append(loss.item())

    return loss_func_relu, model_1


def forward_model_training_2(epochs: int, i: int, lr: float, model: nn.Module, alpha):

    """
    Train the feed-forward neural_network model
    Args:
        epochs
        i: index of the function
        lr: learning rate
        model: The neural network model you want to train the func on
        alpha: The regularization function

    Returns:
        loss_func_relu (list): A list of each loss rate
        model (nn.Module): A list of each loss rate
    """
    # Creating an instance of the model
    model_1 = model
    # Getting the tensors
    Input_train, f_train, Input_test, f_test, f_data, Input_data = data_splitter(i)

    # specifiying criterion and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model_1.parameters(), lr=lr, weight_decay=alpha)

    loss_func_relu = []

    for e in tqdm(range(epochs)):

        # forward pass
        output = model_1(Input_train)
        # compute the loss
        loss = criterion(output, f_train)
        # clear previous gradients
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        optimizer.step()

        loss_func_relu.append(loss.item())

    return loss_func_relu, model_1



def recurrent_model_training_3(epochs: int, i: int, lr: float, model: nn.Module, alpha):

    """
    Train the recurrent neural_network model
    Args:
        epochs
        i: index of the function
        lr: learning rate
        model: The neural network model you want to train the func on
        alpha: The regularization function

    Returns:
        loss_func_relu (list): A list of each loss rate
        model (nn.Module): A list of each loss rate
    """
    # Creating an instance of the model
    model_1 = model
    # Getting the tensors
    Input_train, f_train, Input_test, f_test, f_data, Input_data = data_splitter(i)

    # specifiying criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_1.parameters(), lr=lr, weight_decay=alpha)

    loss_func_relu = []

    for i in tqdm(range(epochs)):

        train = Input_train.reshape(37, 25, -1)
        labels = f_train.reshape(-1, 25)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model_1(train)

        # compute loss
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        loss_func_relu.append(loss.item())

    return loss_func_relu, model_1


def data_splitter(i):

    """
    Split the data and transform it into tensors.
    Args:
        i: index of the function

    Returns:
        Input_train (list): training features,
        f_train (list): target training,
        Input_test (list): testing features,
        f_test (list): target testing,
        Input_data (list): all features,
        f_data (list): all targets,

    """

    # first we read the data for specifid fi function
    if i == 0 or i == 3:
        data_f = pd.read_csv(f"./datasets/f{i}_dataset.csv")
        input_data = data_f["x"].to_numpy()
        f_data = data_f[f"f{i}(x)"].to_numpy()

        data_f = pd.read_csv(f"./datasets/f{i}_dataset.csv")
        f_data = data_f[f"f{i}(x)"].to_numpy()

    elif i == 1 or i == 2:
        data_f = pd.read_csv(f"./datasets/f{i}_dataset.csv")
        input_data = data_f[["x", "y"]].to_numpy()
        f_data = data_f[f"f{i}(x,y)"].to_numpy()

    # splitting the data

    if i == 2:
        Input_train, Input_test, f_train, f_test = train_test_split(
            input_data, f_data, test_size=0.33, random_state=None, shuffle=False
        )
    elif i == 3:
        Input_train, Input_test, f_train, f_test = train_test_split(
            input_data, f_data, test_size=0.075, random_state=None, shuffle=False
        )
    elif i == 1 or i == 0:
        Input_train, Input_test, f_train, f_test = train_test_split(
            input_data, f_data, test_size=0.33, random_state=42
        )

    # converting data to a torch FloatTensor
    if i == 0 or i == 3:
        Input_data = torch.FloatTensor(input_data).reshape(-1, 1)
        f_data = torch.FloatTensor(f_data).reshape(-1, 1)

        Input_train = torch.FloatTensor(Input_train).reshape(-1, 1)
        f_train = torch.FloatTensor(f_train).reshape(-1, 1)

        Input_test = torch.FloatTensor(Input_test).reshape(-1, 1)
        f_test = torch.FloatTensor(f_test).reshape(-1, 1)

    elif i == 1 or i == 2:
        Input_data = torch.FloatTensor(input_data).reshape(-1, 2)

        f_data = torch.FloatTensor(f_data).reshape(-1, 1)

        Input_train = torch.FloatTensor(Input_train).reshape(-1, 2)
        f_train = torch.FloatTensor(f_train).reshape(-1, 1)

        Input_test = torch.FloatTensor(Input_test).reshape(-1, 2)
        f_test = torch.FloatTensor(f_test).reshape(-1, 1)

    return Input_train, f_train, Input_test, f_test, Input_data, f_data
