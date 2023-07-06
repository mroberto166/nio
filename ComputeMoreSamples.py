import os
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
if which == "sine":
    n = 400
    folder = "ImagesSin"

if which == "helm":
    n = 400
    folder = "ImagesHelm"

if not os.path.exists(folder):
    os.makedirs(folder)
pred_vec_1 = torch.zeros((n, b, 70, 70)).to(device)
pred_vec_2 = torch.zeros((n, b, 70, 70)).to(device)
pred_vec_3 = torch.zeros((n, b, 70, 70)).to(device)

if which == "sine" or which == "helm":
    out_vec = torch.zeros((n, b, 70, 70)).to(device)
    shape = (68, 20)
    if which == "sine":
        from Problems.PoissonSin200L import MyDataset as MyDataset1
        from Problems.PoissonSin200L import MyDataset as MyDataset2
        from Problems.PoissonSin200L import MyDataset as MyDataset3
    if which == "helm":
        from Problems.Helm32L import MyDataset as MyDataset1
        from Problems.Helm32L import MyDataset as MyDataset2
        from Problems.Helm32L import MyDataset as MyDataset3
# %%
if which == "sine":
    path1 = "FinalModelNewPerm/Best_nio_new_" + which
    path2 = "FinalModelNewPerm/Best_fcnn_" + which
    path3 = "FinalModelNewPerm/Best_don_" + which

    norm1 = pd.read_csv(path1 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]
    norm2 = pd.read_csv(path2 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]
    norm3 = pd.read_csv(path3 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]
    model1 = torch.load(path1 + "/model.pkl", map_location=torch.device(device))
    model1 = model1.eval()

    model2 = torch.load(path2 + "/model.pkl", map_location=torch.device(device))
    model2 = model2.eval()

    model3 = torch.load(path3 + "/model.pkl", map_location=torch.device(device))
    model3 = model3.eval()

    test_dataset = MyDataset1(norm=norm1, inputs_bool=True, device="cpu", which="testing", mod="nio_new", noise=noise)
    test_dataset_2 = MyDataset2(norm=norm2, inputs_bool=True, device="cpu", which="testing", mod="fcnn", noise=noise)
    test_dataset_3 = MyDataset2(norm=norm3, inputs_bool=True, device="cpu", which="testing", mod="don", noise=noise)

if which == "helm":
    path1 = "FinalModelNewPerm/Best_nio_new_" + which
    path2 = "FinalModelNewPerm/Best_fcnn_" + which
    path3 = "FinalModelNewPerm/Best_don_" + which

    norm1 = pd.read_csv(path1 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]
    norm2 = pd.read_csv(path2 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]
    norm3 = pd.read_csv(path3 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]

    model1 = torch.load(path1 + "/model.pkl", map_location=torch.device('cpu')).cpu()
    model1 = model1.eval()

    model2 = torch.load(path2 + "/model.pkl", map_location=torch.device('cpu')).cpu()
    model2 = model2.eval()

    model3 = torch.load(path3 + "/model.pkl", map_location=torch.device(device))
    model3 = model3.eval()

    test_dataset = MyDataset1(norm=norm1, inputs_bool=True, device=device, which="testing", mod="nio_new", noise=noise)
    test_dataset_2 = MyDataset2(norm=norm2, inputs_bool=True, device=device, which="testing", mod="fcnn", noise=noise)
    test_dataset_3 = MyDataset3(norm=norm3, inputs_bool=True, device=device, which="testing", mod="don", noise=noise)

print("########################################################")
print("NIO params")
model1.print_size()
print("########################################################")
print("FCNN params")
model2.print_size()
print("########################################################")
print("DON params")
model3.print_size()
model1 = model1.to(device)
model2 = model2.to(device)
model3 = model3.to(device)

model1.device = "cpu"

# %%


testing_set = DataLoader(test_dataset, batch_size=b, shuffle=False, num_workers=0, pin_memory=True)
testing_set_2 = DataLoader(test_dataset_2, batch_size=b, shuffle=False, num_workers=0, pin_memory=True)
testing_set_3 = DataLoader(test_dataset_3, batch_size=b, shuffle=False, num_workers=0, pin_memory=True)
if which == "curve" or which == "style":
    grid = test_dataset.get_grid(1, 70).squeeze(0)
else:
    grid = test_dataset.get_grid().squeeze(0)

# %%
########################################################################################
# Evaluation
########################################################################################
errs_vec = np.zeros((n, 3))
errs_vec_2 = np.zeros((n, 3))
errs_vec_3 = np.zeros((n, 3))

running_relative_test_mse = 0.0
running_relative_test_mse_2 = 0.
running_relative_test_mse_3 = 0.
running_ssim = 0.0
min_model = test_dataset.min_model
max_model = test_dataset.max_model

if m != "false":
    np.random.seed(0)
    mmax = 100
    idx = np.random.choice(mmax, int(m), replace=False)
    # idx = np.linspace(0,99,20).astype(int)
    idx_sorted = idx  # [np.argsort(idx)]

# idx_sorted = idx#[np.argsort(idx)]

with torch.no_grad():
    for step, ((input_batch, output_batch), (input_batch_2, _), (input_batch_3, _)) in enumerate(zip(testing_set, testing_set_2, testing_set_3)):
        input_batch = input_batch.to(device)
        output_batch = output_batch.to(device)

        if m != "false":
            if which == "sine" or which == "helm":
                input_batch = input_batch[:, idx_sorted, :, :]
            elif which == "curve" or which == "style":
                input_batch = input_batch[:, :, :, idx_sorted]
            elif which == "heart":
                input_batch = input_batch[:, idx_sorted, :]

            # print(input_batch.shape)

            if which == "heart":
                input_batch_2 = input_batch_2[:, :, idx_sorted, :]
                # input_batch = input_batch[:, :, idx_sorted, :]
                input_batch_3 = input_batch_3[:, :, idx_sorted, :]
            else:
                input_batch_2 = input_batch_2[:, :, :, idx_sorted]
                input_batch_3 = input_batch_3[:, :, :, idx_sorted]
            input_batch_2 = torch.nn.functional.interpolate(input_batch_2, size=shape, mode="nearest")
            input_batch_3 = torch.nn.functional.interpolate(input_batch_3, size=shape, mode="nearest")

        input_batch_2 = input_batch_2.to(device)
        input_batch_3 = input_batch_3.to(device)

        grid = grid.to(device)

        pred_test_1 = 0
        L = 1
        for ll in range(L):
            pred_test_1 = pred_test_1 + model1(input_batch, grid) / L

        if which == "curve" or which == "style":
            input_batch_2 = input_batch_2.permute(0, 3, 1, 2)
            pred_test_2 = model2(input_batch_2).squeeze(1)
            pred_test_3 = model3(input_batch_3, grid.permute(1, 0, 2))
        else:
            pred_test_2 = model2(input_batch_2, grid).squeeze(1)
            pred_test_3 = model3(input_batch_3, grid)
        pred_test_1 = test_dataset.denormalize(pred_test_1)
        pred_test_2 = test_dataset_2.denormalize(pred_test_2)
        pred_test_3 = test_dataset_3.denormalize(pred_test_3)
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
            loss_test_2 = my_loss(pred_test_2, output_batch) / my_loss(torch.zeros_like(output_batch), output_batch)
            err_test_2 = loss_test_2.item() ** (1 / p) * 100
            loss_test_3 = my_loss(pred_test_3, output_batch) / my_loss(torch.zeros_like(output_batch), output_batch)
            err_test_3 = loss_test_3.item() ** (1 / p) * 100

            if p == 1:
                running_relative_test_mse = running_relative_test_mse * step / (step + 1) + err_test / (step + 1)
                running_relative_test_mse_2 = running_relative_test_mse_2 * step / (step + 1) + err_test_2 / (step + 1)
                running_relative_test_mse_3 = running_relative_test_mse_3 * step / (step + 1) + err_test_3 / (step + 1)

            errs_vec[step, p - 1] = err_test
            errs_vec_2[step, p - 1] = err_test_2
            errs_vec_3[step, p - 1] = err_test_3

        pred_vec_1[step, :, :, :] = pred_test_1
        pred_vec_2[step, :, :, :] = pred_test_2
        pred_vec_3[step, :, :, :] = pred_test_3
        out_vec[step, :, :, :] = output_batch

        if step % 1 == 0:
            print("Batch: ", step, running_relative_test_mse, running_relative_test_mse_2, running_relative_test_mse_3)
        if step >= n - 1:
            break

print("Median L1 NIO:", np.median(errs_vec[:, 0]))
print("Median L1 Base:", np.median(errs_vec_2[:, 0]))
print("Median L1 DON:", np.median(errs_vec_3[:, 0]))

print("Median L2 NIO:", np.median(errs_vec[:, 1]))
print("Median L2 Base:", np.median(errs_vec_2[:, 1]))
print("Median L2 DON:", np.median(errs_vec_3[:, 1]))

save_path = "FinalModelNewPerm"
with open(save_path + '/sum_errors_moresam_' + str(noise) + '_' + str(m) + '_' + str(which) + '.txt', 'w') as file:
    file.write("Median L1 NIO:" + str(np.median(errs_vec[:, 0])) + "\n")
    file.write("Median L1 Base:" + str(np.median(errs_vec_2[:, 0])) + "\n")
    file.write("Median L1 DON:" + str(np.median(errs_vec_3[:, 0])) + "\n")

    file.write("25 Quantile L1 NIO:" + str(np.quantile(errs_vec[:, 0], 0.25)) + "\n")
    file.write("25 Quantile L1 Base:" + str(np.quantile(errs_vec_2[:, 0], 0.25)) + "\n")
    file.write("25 Quantile L1 DON:" + str(np.quantile(errs_vec_3[:, 0], 0.25)) + "\n")

    file.write("75 Quantile L1 NIO:" + str(np.quantile(errs_vec[:, 0], 0.75)) + "\n")
    file.write("75 Quantile L1 Base:" + str(np.quantile(errs_vec_2[:, 0], 0.75)) + "\n")
    file.write("75 Quantile L1 DON:" + str(np.quantile(errs_vec_3[:, 0], 0.75)) + "\n")

    file.write("Std L1 NIO:" + str(np.std(errs_vec[:, 0])) + "\n")
    file.write("Std L1 Base:" + str(np.std(errs_vec_2[:, 0])) + "\n")
    file.write("Std L1 DON:" + str(np.std(errs_vec_3[:, 0])) + "\n")

    file.write("Mean L1 NIO:" + str(np.mean(errs_vec[:, 0])) + "\n")
    file.write("Mean L1 Base:" + str(np.mean(errs_vec_2[:, 0])) + "\n")
    file.write("Mean L1 DON:" + str(np.mean(errs_vec_3[:, 0])) + "\n")

    file.write("Median L2 NIO:" + str(np.median(errs_vec[:, 1])) + "\n")
    file.write("Median L2 Base:" + str(np.median(errs_vec_2[:, 1])) + "\n")
    file.write("Median L2 DON:" + str(np.median(errs_vec_3[:, 1])) + "\n")

    file.write("25 Quantile L2 NIO:" + str(np.quantile(errs_vec[:, 1], 0.25)) + "\n")
    file.write("25 Quantile L2 Base:" + str(np.quantile(errs_vec_2[:, 1], 0.25)) + "\n")
    file.write("25 Quantile L2 DON:" + str(np.quantile(errs_vec_3[:, 1], 0.25)) + "\n")

    file.write("75 Quantile L2 NIO:" + str(np.quantile(errs_vec[:, 1], 0.75)) + "\n")
    file.write("75 Quantile L2 Base:" + str(np.quantile(errs_vec_2[:, 1], 0.75)) + "\n")
    file.write("75 Quantile L2 DON:" + str(np.quantile(errs_vec_3[:, 1], 0.75)) + "\n")

    file.write("Std L2 NIO:" + str(np.std(errs_vec[:, 1])) + "\n")
    file.write("Std L2 Base:" + str(np.std(errs_vec_2[:, 1])) + "\n")
    file.write("Std L2 DON:" + str(np.std(errs_vec_3[:, 1])) + "\n")

    file.write("Mean L2 NIO:" + str(np.mean(errs_vec[:, 1])) + "\n")
    file.write("Mean L2 Base:" + str(np.mean(errs_vec_2[:, 1])) + "\n")
    file.write("Mean L2 DON:" + str(np.mean(errs_vec_3[:, 1])) + "\n")
