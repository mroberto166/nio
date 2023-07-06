import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, norm, inputs_bool, device, which, noise=0, mod="nio"):
        self.file_data = "data/StyleData.h5"
        self.noise = noise
        if which == "training":
            self.length = 54999
            self.start = 0
            self.which = which
        elif which == "validation":
            self.length = 5000
            self.start = 54999
            self.which = which
        elif which == "all":
            self.length = 54999 + 5000
            self.start = 0
            self.which = which
        else:
            self.length = 7000
            self.start = 54999 + 5000
            self.which = which
        self.reader = h5py.File(self.file_data, 'r')
        self.mean_inp = torch.from_numpy(self.reader['mean_inp_fun'][:, :]).type(torch.float32)
        self.mean_out = torch.from_numpy(self.reader['mean_out_fun'][:, :]).type(torch.float32)
        self.std_inp = torch.from_numpy(self.reader['std_inp_fun'][:, :]).type(torch.float32)
        self.std_out = torch.from_numpy(self.reader['std_out_fun'][:, :]).type(torch.float32)

        self.min_model = torch.tensor(1515.0234)
        self.max_model = torch.tensor(4446.072265625)

        self.min_data_logt = torch.tensor(-3.256535)
        self.max_data_logt = torch.tensor(3.9105592)

        self.min_data = torch.tensor(-24.95943)
        self.max_data = torch.tensor(48.92686)

        self.inp_dim_branch = 5
        self.n_fun_samples = 5

        self.norm = norm
        self.inputs_bool = inputs_bool

        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader[self.which]['sample_' + str(index + self.start)]["input"][:]).type(torch.float32)
        labels = torch.from_numpy(self.reader[self.which]['sample_' + str(index + self.start)]["output"][:]).type(torch.float32)

        inputs = inputs * (1 + self.noise * torch.randn_like(inputs))
        if self.norm == "norm":
            inputs = self.normalize(inputs, self.mean_inp, self.std_inp)
            labels = self.normalize(labels, self.mean_out, self.std_out)
        elif self.norm == "norm-inp":
            inputs = self.normalize(inputs, self.mean_inp, self.std_inp)
            labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.
        elif self.norm == "log-minmax":
            inputs = (np.log1p(np.abs(inputs))) * np.sign(inputs)
            inputs = 2 * (inputs - self.min_data_logt) / (self.max_data_logt - self.min_data_logt) - 1.
            labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.
        elif self.norm == "minmax":
            inputs = 2 * (inputs - self.min_data) / (self.max_data - self.min_data) - 1.
            labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.
        elif self.norm == "none":
            inputs = inputs
            labels = labels
        else:
            raise ValueError()

        inputs = inputs.permute(1, 2, 0)

        return inputs, labels

    def normalize(self, tensor, mean, std):
        return (tensor - mean) / (std + 1e-16)

    def denormalize(self, tensor):
        if self.norm == "norm":
            return tensor * (self.std_out + 1e-16).to(self.device) + self.mean_out.to(self.device)
        else:
            return (self.max_model - self.min_model) * (tensor + torch.tensor(1., device=self.device)) / 2 + self.min_model.to(self.device)

    def get_grid(self, samples=1, res=70):
        size_x = size_y = res
        samples = samples
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([samples, size_y, 1, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([samples, 1, size_x, 1])
        grid = torch.cat((gridx, gridy), dim=-1).permute(0, 2, 1, 3)

        return grid
