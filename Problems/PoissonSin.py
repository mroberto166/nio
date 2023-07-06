import h5py
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, norm, inputs_bool, device, which, mod, noise=0., samples=4096):
        print("Training with ", samples, " samples")
        self.file_data = "data/Poisson70_L20.h5"
        self.mod = mod
        self.noise = noise
        if which == "training":
            self.length = samples

            self.start = 0
            self.which = which
            print("Using ", self.length, " Training Samples")
        elif which == "validation":
            self.length = 1024
            self.start = 4096
            self.which = which
        else:
            self.length = 2047
            self.start = 4096 + 1024
            self.which = which

        self.reader = h5py.File(self.file_data, 'r')
        self.mean_inp = torch.from_numpy(self.reader['mean_inp_fun'][:, :]).type(torch.float32)
        self.mean_out = torch.from_numpy(self.reader['mean_out_fun'][:, :]).type(torch.float32)
        self.std_inp = torch.from_numpy(self.reader['std_inp_fun'][:, :]).type(torch.float32)
        self.std_out = torch.from_numpy(self.reader['std_out_fun'][:, :]).type(torch.float32)
        self.min_data = self.reader['min_inp'][()]
        self.max_data = self.reader['max_inp'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]
        if self.mod == "nio" or self.mod == "fcnn" or self.mod == "don":
            self.inp_dim_branch = 4
            self.n_fun_samples = 20
        else:
            self.inp_dim_branch = 272
            self.n_fun_samples = 20

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
        elif self.norm == "norm-out":
            inputs = 2 * (inputs - self.min_data) / (self.max_data - self.min_data) - 1.
            labels = self.normalize(labels, self.mean_out, self.std_out)
        elif self.norm == "minmax":
            inputs = 2 * (inputs - self.min_data) / (self.max_data - self.min_data) - 1.
            labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.
        elif self.norm == "none":
            inputs = inputs
            labels = labels
        else:
            raise ValueError()

        if self.mod == "nio" or self.mod == "fcnn" or self.mod == "don":
            inputs = inputs.view(4, 68, 20)
        else:
            inputs = inputs.view(1, 4, 68, 20).permute(3, 0, 1, 2)

        return inputs, labels

    def normalize(self, tensor, mean, std):
        return (tensor - mean) / (std + 1e-16)

    def denormalize(self, tensor):
        if self.norm == "norm" or self.norm == "norm-out":
            return tensor * (self.std_out + 1e-16).to(self.device) + self.mean_out.to(self.device)
        elif self.norm == "none":
            return tensor
        else:
            return (self.max_model - self.min_model) * (tensor + torch.tensor(1., device=self.device)) / 2 + self.min_model

    def get_grid(self):
        grid = torch.from_numpy(self.reader['grid'][:, :]).type(torch.float32)

        return grid.unsqueeze(0)
