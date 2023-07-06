import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_ssim
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
n = 1500
noise = float(sys.argv[1])
m = sys.argv[2]
main_folder = sys.argv[3]
plot = bool(int(sys.argv[4]))

if plot:
    n = 50
pred_vec_1 = np.zeros((n, b, 70))
pred_vec_2 = np.zeros((n, b, 70))
pred_vec_3 = np.zeros((n, b, 70))
out_vec = np.zeros((n, b, 70))
inp_vec = np.zeros((n, b, 32, 32))

from Problems.AlbedoOperator import MyDataset as MyDataset1
from Problems.AlbedoOperator import MyDataset as MyDataset2
from Problems.AlbedoOperator import MyDataset as MyDataset3

# %%
path1 = main_folder + "/Best_nio_new_rad"
path2 = main_folder + "/Best_fcnn_rad"
path3 = main_folder + "/Best_don_rad"
save_path = main_folder

norm1 = pd.read_csv(path1 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]
norm2 = pd.read_csv(path2 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]
norm3 = pd.read_csv(path3 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]

model1 = torch.load(path1 + "/model.pkl", map_location=torch.device('cpu')).cpu()
model1 = model1.eval()

model2 = torch.load(path2 + "/model.pkl", map_location=torch.device('cpu')).cpu()
model2 = model2.eval()

model3 = torch.load(path3 + "/model.pkl", map_location=torch.device('cpu')).cpu()
model3 = model3.eval()

test_dataset = MyDataset1(norm=norm1, inputs_bool=True, device="cpu", which="testing", mod="nio_new", noise=noise)
test_dataset_2 = MyDataset2(norm=norm2, inputs_bool=True, device="cpu", which="testing", mod="fcnn", noise=noise)
test_dataset_3 = MyDataset3(norm=norm3, inputs_bool=True, device="cpu", which="testing", mod="don", noise=noise)

print("########################################################")
print("NIO params")
model1.print_size()
print("########################################################")
print("FCNN params")
model2.print_size()
print("########################################################")
print("DON params")
model3.print_size()

model1.device = "cpu"
model1 = model1.to(device)
model2 = model2.to(device)
model3 = model3.to(device)
# %%
testing_set = DataLoader(test_dataset, batch_size=b, shuffle=False, num_workers=0, pin_memory=True)
testing_set_2 = DataLoader(test_dataset_2, batch_size=b, shuffle=False, num_workers=0, pin_memory=True)
testing_set_3 = DataLoader(test_dataset_3, batch_size=b, shuffle=False, num_workers=0, pin_memory=True)
grid = test_dataset.get_grid().squeeze(0)
# %%
########################################################################################
# Evaluation
########################################################################################
errs_vec = np.zeros((n, 3))
errs_vec_2 = np.zeros((n, 3))
errs_vec_3 = np.zeros((n, 3))

errs_vec = np.zeros((n, 3))
errs_vec_2 = np.zeros((n, 3))
errs_vec_3 = np.zeros((n, 3))

running_relative_test_mse = 0.0
running_relative_test_mse_2 = 0.
running_relative_test_mse_3 = 0.
running_ssim = 0.0
ssim_loss = pytorch_ssim.SSIM(window_size=11)
min_model = test_dataset.min_model
max_model = test_dataset.max_model

if m != "false":
    np.random.seed(0)
    idx = np.random.choice(32, int(m), replace=False)
    idx_sorted = idx  # [np.argsort(idx)]

with torch.no_grad():
    for step, ((input_batch, output_batch), (input_batch_2, _), (input_batch_3, _)) in enumerate(zip(testing_set, testing_set_2, testing_set_3)):
        print(input_batch.shape, input_batch_2.shape, input_batch_3.shape)
        input_batch = input_batch.to(device)
        output_batch = output_batch.to(device)

        input_batch_2 = input_batch_2.to(device)

        input_batch_3 = input_batch_3.to(device)

        if m != "false":
            input_batch = input_batch[:, idx_sorted, :]
            input_batch_2 = input_batch_2[:, :, :, idx_sorted]
            input_batch_3 = input_batch_3[:, :, :, idx_sorted]

            # print(input_batch.shape)

            # input_batch = torch.nn.functional.interpolate(input_batch, size=(32,32), mode="nearest")
            input_batch_2 = torch.nn.functional.interpolate(input_batch_2, size=(32, 32), mode="nearest")
            input_batch_3 = torch.nn.functional.interpolate(input_batch_3, size=(32, 32), mode="nearest")

        grid = grid.to(device)

        # input_batch = input_batch * (1 + noise * torch.randn_like(input_batch, device=device))
        # input_batch_2 = input_batch_2 * (1 + noise * torch.randn_like(input_batch, device=device))
        # input_batch_3 = input_batch_3 * (1 + noise * torch.randn_like(input_batch, device=device))

        pred_test_1 = model1(input_batch, grid)
        pred_test_2 = model2(input_batch_2, grid).squeeze(1)
        pred_test_3 = model3(input_batch_3, grid)
        pred_test_1 = test_dataset.denormalize(pred_test_1)
        pred_test_2 = test_dataset_2.denormalize(pred_test_2)
        pred_test_3 = test_dataset_3.denormalize(pred_test_3)
        output_batch = test_dataset.denormalize(output_batch)

        # np.savetxt("data_devito/exact_vel_" + str(step)+".txt", output_batch[0].detach().numpy())
        # np.savetxt("data_devito/pred_vel_NIO_" + str(step)+".txt", pred_test_1[0].detach().numpy())
        # np.savetxt("data_devito/pred_vel_Inv_" + str(step)+".txt", pred_test_2[0].detach().numpy())

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

        pred_vec_1[step, :, :] = pred_test_1
        pred_vec_2[step, :, :] = pred_test_2
        pred_vec_3[step, :, :] = pred_test_3
        out_vec[step, :, :] = output_batch
        inp_vec[step, :, :, :] = input_batch_3

        # output_batch = 2 * (output_batch - min_model) / (max_model - min_model) - 1.
        # pred_test_1 = 2 * (pred_test_1 - min_model) / (max_model - min_model) - 1.
        # pred_test_2 = 2 * (pred_test_2 - min_model) / (max_model - min_model) - 1.
        # pred_test_3 = 2 * (pred_test_3 - min_model) / (max_model - min_model) - 1.

        # ssim_1 = ssim_loss(pred_test_1.unsqueeze(1) / 2 + 0.5, output_batch.unsqueeze(1) / 2 + 0.5).item()
        # ssim_2 = ssim_loss(pred_test_2.unsqueeze(1) / 2 + 0.5, output_batch.unsqueeze(1) / 2 + 0.5).item()
        # ssim_3 = ssim_loss(pred_test_3.unsqueeze(1) / 2 + 0.5, output_batch.unsqueeze(1) / 2 + 0.5).item()
        # errs_vec[step, 2] = ssim_1
        # errs_vec_2[step, 2] = ssim_2
        # errs_vec_3[step, 2] = ssim_3
        # ssim = ssim_loss(pred_test.unsqueeze(0), output_batch.unsqueeze(0)).item()
        # running_ssim = running_ssim * step / (step + 1) + ssim_1 / (step + 1)

        if step % 1 == 0:
            print("Batch: ", step, running_relative_test_mse, running_relative_test_mse_2, running_relative_test_mse_3)
        if step >= n - 1:
            break

if not plot:
    with open(save_path + '/sum_errors_complete_' + str(noise) + '_' + str(m) + "_" + str("rad") + '.txt', 'w') as file:
        file.write("Median L1 NIO:" + str(np.median(errs_vec[:, 0])) + "\n")
        file.write("Median L1 FCNN:" + str(np.median(errs_vec_2[:, 0])) + "\n")
        file.write("Median L1 DON:" + str(np.median(errs_vec_3[:, 0])) + "\n")

        file.write("25 Quantile L1 NIO:" + str(np.quantile(errs_vec[:, 0], 0.25)) + "\n")
        file.write("25 Quantile L1 FCNN:" + str(np.quantile(errs_vec_2[:, 0], 0.25)) + "\n")
        file.write("25 Quantile L1 DON:" + str(np.quantile(errs_vec_3[:, 0], 0.25)) + "\n")

        file.write("75 Quantile L1 NIO:" + str(np.quantile(errs_vec[:, 0], 0.75)) + "\n")
        file.write("75 Quantile L1 FCNN:" + str(np.quantile(errs_vec_2[:, 0], 0.75)) + "\n")
        file.write("75 Quantile L1 DON:" + str(np.quantile(errs_vec_3[:, 0], 0.75)) + "\n")

        file.write("Std L1 NIO:" + str(np.std(errs_vec[:, 0])) + "\n")
        file.write("Std L1 FCNN:" + str(np.std(errs_vec_2[:, 0])) + "\n")
        file.write("Std L1 DON:" + str(np.std(errs_vec_3[:, 0])) + "\n")

        file.write("Mean L1 NIO:" + str(np.mean(errs_vec[:, 0])) + "\n")
        file.write("Mean L1 FCNN:" + str(np.mean(errs_vec_2[:, 0])) + "\n")
        file.write("Mean L1 DON:" + str(np.mean(errs_vec_3[:, 0])) + "\n")

        file.write("Median L2 NIO:" + str(np.median(errs_vec[:, 1])) + "\n")
        file.write("Median L2 FCNN:" + str(np.median(errs_vec_2[:, 1])) + "\n")
        file.write("Median L2 DON:" + str(np.median(errs_vec_3[:, 1])) + "\n")

        file.write("25 Quantile L2 NIO:" + str(np.quantile(errs_vec[:, 1], 0.25)) + "\n")
        file.write("25 Quantile L2 FCNN:" + str(np.quantile(errs_vec_2[:, 1], 0.25)) + "\n")
        file.write("25 Quantile L2 DON:" + str(np.quantile(errs_vec_3[:, 1], 0.25)) + "\n")

        file.write("75 Quantile L2 NIO:" + str(np.quantile(errs_vec[:, 1], 0.75)) + "\n")
        file.write("75 Quantile L2 FCNN:" + str(np.quantile(errs_vec_2[:, 1], 0.75)) + "\n")
        file.write("75 Quantile L2 DON:" + str(np.quantile(errs_vec_3[:, 1], 0.75)) + "\n")

        file.write("Std L2 NIO:" + str(np.std(errs_vec[:, 1])) + "\n")
        file.write("Std L2 FCNN:" + str(np.std(errs_vec_2[:, 1])) + "\n")
        file.write("Std L2 DON:" + str(np.std(errs_vec_3[:, 1])) + "\n")

        file.write("Mean L2 NIO:" + str(np.mean(errs_vec[:, 1])) + "\n")
        file.write("Mean L2 FCNN:" + str(np.mean(errs_vec_2[:, 1])) + "\n")
        file.write("Mean L2 DON:" + str(np.mean(errs_vec_3[:, 1])) + "\n")
else:
    # %%
    p = 1

    for i in range(0, n):
        pred_test_i_1 = torch.tensor(pred_vec_1[i, 0, :]) - 1
        pred_test_i_2 = torch.tensor(pred_vec_2[i, 0, :]) - 1
        labels_i = torch.tensor(out_vec[i, 0, :]) - 1

        vmax = torch.max(labels_i)
        vmin = torch.min(labels_i)
        # out_deep = (deeponet(inp_deep, grid_deeponet).reshape(70, 70))

        # err_stack = ((torch.mean(torch.abs(labels_i - pred_test_i_stack) ** p)
        #              / torch.mean(torch.abs(labels_i) ** p)) ** (1 / p)).item() * 100
        err_1 = ((torch.mean(torch.abs(labels_i - pred_test_i_1) ** p) / torch.mean(torch.abs(labels_i) ** p)) ** (1 / p)).item() * 100
        err_2 = ((torch.mean(torch.abs(labels_i - pred_test_i_2) ** p) / torch.mean(torch.abs(labels_i) ** p)) ** (1 / p)).item() * 100
        err_1 = round(err_1, 1)
        err_2 = round(err_2, 1)
        print(i, err_1, err_2)
        fig = plt.figure(dpi=200)
        plt.grid(True, which="both", ls=":")
        plt.plot(grid, labels_i + 1, label=r'Exact Coefficient', c="grey", lw=4)
        plt.scatter(grid, pred_test_i_1 + 1, label=r'NIO')
        plt.scatter(grid, pred_test_i_2 + 1, label=r'FCNN')

        plt.ylim(0.8, 2.2)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$a(x)$')
        plt.legend()
        plt.savefig("ImagesRadNew/Sample" + str(i) + "_rad.png", dpi=200)
