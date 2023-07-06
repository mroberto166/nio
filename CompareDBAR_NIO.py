import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from Problems.HeartLungsEIT import MyDataset as MyDataset1
from scipy.interpolate import griddata


def polar(x_, y_):
    rho = np.sqrt(x_ ** 2 + y_ ** 2)
    phi = np.arctan2(y_, x_)
    return rho, phi


# find the polar coordinates of complex number
noise = 0
device = "cpu"
path1 = "FinalModelNewPerm/Best_nio_new_eit"
save_path = "FinalModelNewPerm/"

norm1 = pd.read_csv(path1 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]

model1 = torch.load(path1 + "/model.pkl", map_location=torch.device(device))
model1 = model1.eval()

test_dataset = MyDataset1(norm=norm1, inputs_bool=True, device="cpu", which="testing", mod="nio_new", noise=noise)

folder = "data/data_8000"

for folder in ["data/data_8000", "data/data_9000"]:
    print(folder)
    x = np.loadtxt(folder + "/x.txt")
    y = np.loadtxt(folder + "/y.txt")
    cond = np.loadtxt(folder + "/cond.txt")
    dtn_1 = np.loadtxt(folder + "/dn_real.txt")
    dtn = np.loadtxt(folder + "/dn1_real.txt")

    data = (dtn - dtn_1).reshape(33, 33)

    r, theta = polar(x, y)

    n = 70

    r_grid = np.linspace(min(r), max(r), n)
    theta_grid = np.linspace(-1. * np.pi, 1. * np.pi, n)
    rr, tt = np.meshgrid(r_grid, theta_grid)

    cond = griddata(np.concatenate((r.reshape(-1, 1), theta.reshape(-1, 1)), -1), cond, (rr, tt), method='nearest', fill_value=0.)

    grid = np.concatenate((tt.reshape(tt.shape[0], tt.shape[1], 1), rr.reshape(rr.shape[0], rr.shape[1], 1)), -1)
    data = np.delete(data, [16], axis=0)
    data = np.delete(data, [16], axis=1)
    grid = torch.tensor(grid).type(torch.float32)
    data = torch.tensor(data).type(torch.float32)
    data = 2 * (data - test_dataset.min_data) / (test_dataset.max_data - test_dataset.min_data) - 1.
    data = data.reshape(1, 32, 32)

    start = time.time()
    out = model1(data, grid)
    outputs = test_dataset.denormalize(out)
    end = time.time()

    elapsed = end - start

    print("inference time ", elapsed)

    x = np.loadtxt(folder + "/x1.txt")
    y = np.loadtxt(folder + "/x2.txt")
    recon_dbar = np.loadtxt(folder + "/recon.txt")

    r, theta = polar(x, y)

    n = 70

    r_grid = np.linspace(0, 0.95, n)
    theta_grid = np.linspace(-np.pi, np.pi, n)
    rr, tt = np.meshgrid(r_grid, theta_grid)

    recon_dbar = griddata(np.concatenate((r.reshape(-1, 1), theta.reshape(-1, 1)), -1), recon_dbar.reshape(-1, 1), (rr, tt), method='nearest', fill_value=0.)

    grid = np.concatenate((tt.reshape(tt.shape[0], tt.shape[1], 1), rr.reshape(rr.shape[0], rr.shape[1], 1)), -1)

    recon_dbar = recon_dbar[:, :, 0]

    dbar_error = np.mean(abs(recon_dbar - cond)) / np.mean(abs(cond)) * 100
    nio_error = np.mean(abs(outputs[0].detach().numpy() - cond)) / np.mean(abs(cond)) * 100

    print("DBar Error ", dbar_error)
    print("NIO Error ", nio_error)

    fig, ax = plt.subplots(1, 3, figsize=(30, 8), subplot_kw=dict(projection='polar'))
    ax[0].grid(False)
    ax[1].grid(False)
    ax[2].grid(False)
    ax[0].contourf(grid[:, :, 0], grid[:, :, 1], cond, levels=50, cmap="inferno", vmin=test_dataset.min_model, vmax=2)
    ax[0].set_yticklabels([])
    ax[0].set_theta_zero_location("W")

    ax[1].contourf(grid[:, :, 0], grid[:, :, 1], outputs[0].detach(), levels=50, cmap="inferno", vmin=test_dataset.min_model, vmax=2)
    ax[1].set_yticklabels([])
    ax[1].set_theta_zero_location("W")

    im3 = ax[2].contourf(grid[:, :, 0], grid[:, :, 1], recon_dbar, levels=50, cmap="inferno", vmin=test_dataset.min_model, vmax=2)
    ax[2].set_yticklabels([])
    ax[2].set_theta_zero_location("W")

    ax[0].set(title="Exact Coefficient")
    ax[1].set(title="NIO Predicted Coefficient")
    ax[2].set(title="DBAR Method Reconstruction")

    cax, kw = mpl.colorbar.make_axes([a for a in ax.flat])
    plt.colorbar(im3, cax=cax, **kw)

    plt.savefig(save_path + "/" + folder.split("/")[1] + "dbar_comp.png", dpi=200, bbox_inches='tight')
