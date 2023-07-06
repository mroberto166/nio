import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

np.random.seed(42)
random.seed(42)
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the tick labels
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)
print(torch.__version__)
print(sys.version)

b = 1
which = sys.argv[1]
noise = float(sys.argv[2])
m = sys.argv[3]
main_folder = sys.argv[4]

n = 1000

if which == "curve":
    from Problems.CurveVel import MyDataset as MyDataset1

    out_vec = torch.zeros((n, b, 70, 70)).to(device)
    inp_vec = torch.zeros((n, b, 1000, 70, 5)).to(device)
    shape = (70, 5)
if which == "style":
    from Problems.StlyleData import MyDataset as MyDataset1

if which == "sine" or which == "step" or which == "helm":
    if which == "sine":
        from Problems.PoissonSin import MyDataset as MyDataset1
    if which == "helm":
        from Problems.HelmNIO import MyDataset as MyDataset1
if which == "eit":
    from Problems.HeartLungsEIT import MyDataset as MyDataset1
if which == "rad":
    from Problems.AlbedoOperator import MyDataset as MyDataset1
# %%


if which == "sine":
    mmax = 20
    path1 = main_folder + "/Best_nio_new_" + which

    norm1 = pd.read_csv(path1 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]

    model1 = torch.load(path1 + "/model.pkl", map_location=torch.device(device))
    model1 = model1.eval()

    test_dataset = MyDataset1(norm=norm1, inputs_bool=True, device="cpu", which="testing", mod="nio_new", noise=noise)

if which == "eit":
    mmax = 32
    path1 = main_folder + "/Best_nio_new_" + which

    norm1 = pd.read_csv(path1 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]

    model1 = torch.load(path1 + "/model.pkl", map_location=torch.device(device))
    model1 = model1.eval()

    test_dataset = MyDataset1(norm=norm1, inputs_bool=True, device="cpu", which="testing", mod="nio_new", noise=noise)

if which == "helm":
    mmax = 20
    path1 = main_folder + "/Best_nio_new_" + which

    norm1 = pd.read_csv(path1 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]

    model1 = torch.load(path1 + "/model.pkl", map_location=torch.device('cpu')).cpu()
    model1 = model1.eval()

    test_dataset = MyDataset1(norm=norm1, inputs_bool=True, device=device, which="testing", mod="nio_new", noise=noise)

if which == "rad":
    mmax = 32
    path1 = main_folder + "/Best_nio_new_" + which

    norm1 = pd.read_csv(path1 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]

    model1 = torch.load(path1 + "/model.pkl", map_location=torch.device('cpu')).cpu()
    model1 = model1.eval()

    test_dataset = MyDataset1(norm=norm1, inputs_bool=True, device=device, which="testing", mod="nio_new", noise=noise)

print("########################################################")
print("NIO params")
model1.print_size()
print("########################################################")
model1 = model1.to(device)

model1.device = "cpu"

testing_set = DataLoader(test_dataset, batch_size=b, shuffle=False, num_workers=0, pin_memory=True)
grid = test_dataset.get_grid().squeeze(0)

# %%
########################################################################################
# Evaluation
########################################################################################
errs_vec = np.zeros((n, 3))

running_relative_test_mse = 0.0
running_relative_test_mse_2 = 0.
running_relative_test_mse_3 = 0.
min_model = test_dataset.min_model
max_model = test_dataset.max_model

if m != "false":
    np.random.seed(0)

    idx = np.random.choice(mmax, int(m), replace=False)
    idx_sorted = idx  # [np.argsort(idx)]

with torch.no_grad():
    for step, (input_batch, output_batch) in enumerate(testing_set):
        input_batch = input_batch.to(device)
        output_batch = output_batch.to(device)

        if m != "false":
            if which == "sine" or which == "helm":
                input_batch = input_batch[:, idx_sorted, :, :]
            elif which == "eit" or which == "rad":
                input_batch = input_batch[:, idx_sorted, :]

        grid = grid.to(device)

        pred_test_1 = 0
        L = 1
        for ll in range(L):
            pred_test_1 = pred_test_1 + model1(input_batch, grid) / L

        pred_test_1 = test_dataset.denormalize(pred_test_1)

        output_batch = test_dataset.denormalize(output_batch)

        for p in [1, 2]:
            if p == 2:
                my_loss = torch.nn.MSELoss()
            elif p == 1:
                my_loss = torch.nn.L1Loss()
            else:
                raise ValueError("Choose p = 1 or p=2")
            loss_test = my_loss(pred_test_1, output_batch) / my_loss(torch.zeros_like(output_batch), output_batch)
            err_test = loss_test.item() ** (1 / p) * 100

            if p == 1:
                running_relative_test_mse = running_relative_test_mse * step / (step + 1) + err_test / (step + 1)

            errs_vec[step, p - 1] = err_test

        if step % 1 == 0:
            print("Batch: ", step, running_relative_test_mse, running_relative_test_mse_2, running_relative_test_mse_3)
        if step >= n - 1:
            break
print("Median L1 NIO:", np.median(errs_vec[:, 0]))
print("Median L2 NIO:", np.median(errs_vec[:, 1]))

save_path = main_folder
print("save in ", save_path + '/sum_errors_abl_' + str(noise) + '_' + str(m) + "_" + str(which) + '.txt')
with open(save_path + '/sum_errors_abl_' + str(noise) + '_' + str(m) + "_" + str(which) + '.txt', 'w') as file:
    file.write("Median L1 NIO:" + str(np.median(errs_vec[:, 0])) + "\n")
    file.write("25 Quantile L1 NIO:" + str(np.quantile(errs_vec[:, 0], 0.25)) + "\n")
    file.write("75 Quantile L1 NIO:" + str(np.quantile(errs_vec[:, 0], 0.75)) + "\n")
    file.write("Std L1 NIO:" + str(np.std(errs_vec[:, 0])) + "\n")
    file.write("Mean L1 NIO:" + str(np.mean(errs_vec[:, 0])) + "\n")
    file.write("Median L2 NIO:" + str(np.median(errs_vec[:, 1])) + "\n")
    file.write("25 Quantile L2 NIO:" + str(np.quantile(errs_vec[:, 1], 0.25)) + "\n")
    file.write("75 Quantile L2 NIO:" + str(np.quantile(errs_vec[:, 1], 0.75)) + "\n")
    file.write("Std L2 NIO:" + str(np.std(errs_vec[:, 1])) + "\n")
    file.write("Mean L2 NIO:" + str(np.mean(errs_vec[:, 1])) + "\n")
