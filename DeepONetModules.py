import numpy as np
import torch.nn as nn

from debug_tools import *


def kaiming_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight.data, a=0.01, nonlinearity='leaky_relu')
        torch.nn.init.zeros_(m.bias.data)


class FourierFeatures(nn.Module):

    def __init__(self, scale, mapping_size, device):
        super().__init__()
        self.mapping_size = mapping_size
        self.B = scale * torch.randn((self.mapping_size, 2)).to(device)

    def forward(self, x):
        x_proj = torch.matmul((2. * np.pi * x), self.B.T)
        inp = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        return inp


class Swish(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Sin(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['leaky_relu']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['elu']:
        return nn.ELU()
    elif name in ['swish']:
        return Swish()
    elif name in ['mish']:
        return nn.Mish()
    elif name in ['sin']:
        return Sin()
    else:
        raise ValueError('Unknown activation function')


def init_xavier(model):
    torch.manual_seed(model.retrain)

    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            if model.act_string == "tanh" or model.act_string == "relu" or model.act_string == "leaky_relu":
                gain = nn.init.calculate_gain(model.act_string)
            else:
                gain = 1
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0)

    model.apply(init_weights)


class FeedForwardNN(nn.Module):

    def __init__(self, input_dimension, output_dimension, network_architecture):
        super(FeedForwardNN, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = network_architecture["n_hidden_layers"]
        self.neurons = network_architecture["neurons"]
        self.act_string = network_architecture["act_string"]
        self.retrain = network_architecture["retrain"]
        self.dropout_rate = network_architecture["dropout_rate"]

        torch.manual_seed(self.retrain)

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.batch_layers = nn.ModuleList(
            [nn.BatchNorm1d(self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.activation = activation(self.act_string)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.apply(kaiming_init)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for k, (l, b) in enumerate(zip(self.hidden_layers, self.batch_layers)):
            x = b(self.activation(self.dropout(l(x))))
        return self.output_layer(x)

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class DeepOnetNoBiasOrg(nn.Module):
    def __init__(self, branch, trunk):
        super(DeepOnetNoBiasOrg, self).__init__()
        self.branch = branch
        self.trunk = trunk
        self.b0 = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.p = self.trunk.output_dimension
        '''self.b0 = torch.nn.Sequential(nn.Linear(self.trunk.input_dimension, 128),
                                      nn.LeakyReLU(),
                                      nn.Linear(128, 128),
                                      nn.LeakyReLU(),
                                      nn.Linear(128, 1))'''
        # self.b0.apply(kaiming_init)

    def forward(self, u_, x_):
        # print(x_.shape)
        weights = self.branch(u_)
        basis = self.trunk(x_)
        # basis = self.trunk(x_)
        # bias = self.b0(x_).squeeze(1)
        # print(bias.shape)
        # print(torch.matmul(weights, basis.T).shape)
        # return (torch.matmul(weights, basis.T) + self.b0)/self.p + mean
        return (torch.matmul(weights, basis.T) + self.b0) / self.p ** 0.5


class KappaOpt(nn.Module):

    def __init__(self, network_architecture):
        super(KappaOpt, self).__init__()
        self.input_dimension = 2
        self.n_hidden_layers = network_architecture["n_hidden_layers"]
        self.neurons = network_architecture["neurons"]
        self.act_string = network_architecture["act_string"]
        self.retrain = network_architecture["retrain"]
        self.dropout_rate = network_architecture["dropout_rate"]

        torch.manual_seed(self.retrain)

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.batch_layers = nn.ModuleList(
            [nn.BatchNorm1d(self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, 1)

        self.activation = activation(self.act_string)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.apply(kaiming_init)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(-1)
        x2 = x2.unsqueeze(-1)

        x = torch.cat((x1, x2), -1)
        x = self.activation(self.input_layer(x))
        for k, (l, b) in enumerate(zip(self.hidden_layers, self.batch_layers)):
            x = self.activation(self.dropout(l(x)))
        x = self.output_layer(x).reshape(x1.squeeze(-1).shape, )
        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams
