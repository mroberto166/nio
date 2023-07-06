import json
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from DeepONetModules import KappaOpt
from SolveHelmTorch import solve_helm

L = int(sys.argv[1])
folder = sys.argv[2]

writer = SummaryWriter(log_dir=folder)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

data = torch.tensor(np.load("data/Opt/input_boundary30.npy")).to(device).squeeze(1).reshape(20, 272).permute(1, 0)
exact_conduct = torch.tensor(np.load("data/Opt/exact30.npy"))

network_architecture = json.loads(sys.argv[3].replace("\'", "\""))
print(network_architecture)

kappa = KappaOpt(network_architecture)

optimizer = torch.optim.LBFGS(kappa.parameters(),
                              lr=float(network_architecture["lr"]),
                              max_iter=50000,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)

running_loss = list()

iteration = 0

x = torch.linspace(0, 1, 70)
y = torch.linspace(0, 1, 70)

x_, y_ = torch.meshgrid(x, y)

p = 2


# Loop over batches
def closure():
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    sim_data, out = solve_helm(kappa, L)
    # Item 1. below
    loss = torch.mean(abs(sim_data - data) ** p) / torch.mean(abs(data) ** p)
    err_cond = (torch.mean((exact_conduct - out) ** 2) / torch.mean(exact_conduct ** 2)) ** 0.5

    # Item 2. below
    loss.backward()
    global iteration
    iteration = iteration + 1
    print(iteration, loss, err_cond)
    writer.add_scalar("Loss", loss, iteration)
    writer.add_scalar("Err", err_cond, iteration)
    if iteration % 5 == 0:
        print("Saving")
        torch.save(kappa, folder + "/model.pkl")

    # Compute average training loss over batches for the current epoch
    return loss


optimizer.step(closure=closure)
writer.flush()
