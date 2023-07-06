import os
import random
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import network

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
plot = bool(int(sys.argv[5]))
if which == "style":
    n = 5500
    folder = "ImagesStyleNew"
if which == "curve":
    n = 6000
    folder = "ImagesCurveNew"
if which == "step" or which == "sine":
    n = 2048
    folder = "ImagesSinNew2"
if which == "eit":
    n = 1900
    folder = "ImagesHeartNew"
if "helm" in which:
    n = 1620
    folder = "ImagesHelmNew2"
if plot:
    n = 1
if not os.path.exists(folder):
    os.makedirs(folder)
pred_vec_1 = torch.zeros((n, b, 70, 70)).to(device)
pred_vec_2 = torch.zeros((n, b, 70, 70)).to(device)
pred_vec_3 = torch.zeros((n, b, 70, 70)).to(device)
if which == "curve":
    from Problems.CurveVel import MyDataset as MyDataset1
    from Problems.CurveVel import MyDataset as MyDataset2
    from Problems.CurveVel import MyDataset as MyDataset3

    out_vec = torch.zeros((n, b, 70, 70)).to(device)
    inp_vec = torch.zeros((n, b, 1000, 70, 5)).to(device)
    shape = (70, 5)
if which == "style":
    from Problems.StlyleData import MyDataset as MyDataset1
    from Problems.StlyleData import MyDataset as MyDataset2
    from Problems.StlyleData import MyDataset as MyDataset3

    out_vec = torch.zeros((n, b, 70, 70)).to(device)
    inp_vec = torch.zeros((n, b, 1000, 70, 5)).to(device)
    shape = (70, 5)

if which == "sine" or which == "step" or "helm" in which:
    out_vec = torch.zeros((n, b, 70, 70)).to(device)
    inp_vec = torch.zeros((n, b, 4, 68, 20)).to(device)
    shape = (68, 20)
    if which == "sine":
        from Problems.PoissonSin import MyDataset as MyDataset1
        from Problems.PoissonSin import MyDataset as MyDataset2
        from Problems.PoissonSin import MyDataset as MyDataset3
    if "helm" in  which:
        from Problems.HelmNIO import MyDataset as MyDataset1
        from Problems.HelmNIO import MyDataset as MyDataset2
        from Problems.HelmNIO import MyDataset as MyDataset3
if which == "eit":
    from Problems.HeartLungsEIT import MyDataset as MyDataset1
    from Problems.HeartLungsEIT import MyDataset as MyDataset2
    from Problems.HeartLungsEIT import MyDataset as MyDataset3

    out_vec = torch.zeros((n, b, 70, 70)).to(device)
    inp_vec = torch.zeros((n, b, 1, 32, 32)).to(device)
    shape = (32, 32)
# %%
if which == "curve" or which == "style":
    path1 = main_folder + "/Best_nio_new_" + which
    path2 = main_folder + "/Best_fcnn_" + which
    path3 = main_folder + "/Best_don_" + which

    norm1 = pd.read_csv(path1 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]
    norm2 = "log-minmax"
    norm3 = pd.read_csv(path3 + "/training_properties.txt", header=None, sep=",", index_col=0).transpose().reset_index().drop("index", 1)["norm"][0]

    model1 = torch.load(path1 + "/model.pkl", map_location=torch.device(device))
    model1 = model1.eval()

    model2 = network.model_dict["InversionNet"](upsample_mode=None,
                                                sample_spatial=1, sample_temporal=1)

    checkpoint = torch.load(path2 + "/checkpoint.pth", map_location=torch.device(device))
    model2.load_state_dict(network.replace_legacy(checkpoint['model']))
    model2 = model2.eval()

    model3 = torch.load(path3 + "/model.pkl", map_location=torch.device(device))
    model3 = model3.eval()

    test_dataset = MyDataset1(norm=norm1, inputs_bool=True, device=device, which="testing", noise=noise, mod="nio_new")
    test_dataset_2 = MyDataset2(norm=norm2, inputs_bool=True, device=device, which="testing", noise=noise)
    test_dataset_3 = MyDataset3(norm=norm3, inputs_bool=True, device=device, which="testing", noise=noise)

    print(norm1, norm2, norm3)

if which == "sine":
    path1 = main_folder + "/Best_nio_new_" + which
    path2 = main_folder + "/Best_fcnn_" + which
    path3 = main_folder + "/Best_don_" + which

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

if which == "eit":
    path1 = main_folder + "/Best_nio_new_" + which
    path2 = main_folder + "/Best_fcnn_" + which
    path3 = main_folder + "/Best_don_" + which

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

if "helm" in  which:
    path1 = main_folder + "/Best_nio_new_helm"
    if which=="helm_stab":
        path1 = main_folder + "/Best_nio_new_helm_stab"
    path2 = main_folder + "/Best_fcnn_helm"
    path3 = main_folder + "/Best_don_helm"

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

testing_set = DataLoader(test_dataset, batch_size=b, shuffle=False, num_workers=0, pin_memory=True)
testing_set_2 = DataLoader(test_dataset_2, batch_size=b, shuffle=False, num_workers=0, pin_memory=True)
testing_set_3 = DataLoader(test_dataset_3, batch_size=b, shuffle=False, num_workers=0, pin_memory=True)
if which == "curve" or which == "style":
    grid = test_dataset.get_grid(1, 70).squeeze(0)
    print(grid.shape)
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
min_model = test_dataset.min_model
max_model = test_dataset.max_model

if m != "false":
    np.random.seed(0)
    idx = np.random.choice(int(shape[-1]), int(m), replace=False)
    idx_sorted = idx  # [np.argsort(idx)]
# idx_sorted = idx#[np.argsort(idx)]

with torch.no_grad():
    for step, ((input_batch, output_batch), (input_batch_2, _), (input_batch_3, _)) in enumerate(zip(testing_set, testing_set_2, testing_set_3)):
        input_batch = input_batch.to(device)
        output_batch = output_batch.to(device)

        print(input_batch.shape, input_batch_2.shape, input_batch_3.shape)

        if m != "false":
            if which == "sine" or which == "helm":
                input_batch = input_batch[:, idx_sorted, :, :]
            elif which == "curve" or which == "style":
                input_batch = input_batch[:, :, :, idx_sorted]
            elif which == "eit":
                input_batch = input_batch[:, idx_sorted, :]

            # print(input_batch.shape)

            if which == "eit":
                input_batch_2 = input_batch_2[:, :, idx_sorted, :]
                # input_batch = input_batch[:, :, idx_sorted, :]
                input_batch_3 = input_batch_3[:, :, idx_sorted, :]
            else:
                input_batch_2 = input_batch_2[:, :, :, idx_sorted]
                input_batch_3 = input_batch_3[:, :, :, idx_sorted]
            # input_batch = torch.nn.functional.interpolate(input_batch, size=shape, mode="nearest")
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
            pred_test_3 = model3(input_batch_3, grid)
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
        inp_vec[step, :, :, :] = input_batch_3  # .reshape(1,4,68,20)

        if step % 1 == 0:
            print("Batch: ", step, running_relative_test_mse, running_relative_test_mse_2, running_relative_test_mse_3)
        if step >= n - 1:
            break

save_path = main_folder
if not plot:
    with open(save_path + '/sum_errors_complete_' + str(noise) + '_' + str(m) + "_" + str(which) + '.txt', 'w') as file:
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
    idx_sorted = np.argsort(errs_vec[:, 0])

    errs_vec_sorted = errs_vec[idx_sorted]
    pred_vec_sorted = pred_vec_1[idx_sorted]
    out_vec_sorted = out_vec[idx_sorted]
    inp_vec_sorted = inp_vec[idx_sorted]
    print(inp_vec_sorted.shape)
    deeponet = model1.deeponet
    nx = (grid.shape[0])
    ny = (grid.shape[1])
    dim = (grid.shape[2])
    grid_deeponet = grid.reshape(-1, dim)
    # %%
    p = 1
    if which == "curve":
        vmax = max_model
    elif which == "sin":
        vmax = None

    pw_errs_1 = torch.tensor((abs(out_vec - pred_vec_1) / abs(out_vec))).squeeze(1)
    pw_errs_2 = torch.tensor((abs(out_vec - pred_vec_2) / abs(out_vec))).squeeze(1)

    max_err = max(torch.max(pw_errs_1), torch.max(pw_errs_2)).item()
    min_err = min(torch.min(pw_errs_1), torch.min(pw_errs_2)).item()

    print(max_err, min_err)

    for i in range(0, n):
        pred_test_i_1 = torch.tensor(pred_vec_1[i, 0, :, :])
        pred_test_i_2 = torch.tensor(pred_vec_2[i, 0, :, :])
        labels_i = torch.tensor(out_vec[i, 0, :, :])

        if which == "helm" or which == "sine":

            bnet = deeponet.branch
            for name, param in bnet.named_parameters():
                if "weight" in name and "0" in name:
                    fig, axes = plt.subplots(1, 3)
                    axes[0].imshow(param.data[..., 0, 0], aspect="auto")
                    axes[1].imshow(param.data[..., 0, 1], aspect="auto")
                    axes[2].imshow(param.data[..., 0, 2], aspect="auto")
                    plt.savefig(folder + "/" + name + ".png", dpi=200)
            inp_deep = torch.tensor(inp_vec[i, :, :, :]).type(torch.float32)
            inp_deep = inp_deep.permute(3, 0, 1, 2).unsqueeze(0)
            out_deep = deeponet(inp_deep, grid_deeponet)
            out_deep = out_deep.reshape(out_deep.shape[0], out_deep.shape[1], nx, ny)[0]
            inp_deep = inp_deep[0].reshape(20, -1)
            plt.figure()

            rows = 4
            columns = 5

            fig1, axes1 = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
            fig2, axes2 = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))

            for index in range(out_deep.shape[0]):
                q = index // 5
                mod = index % 5
                axes1[q, mod].plot(np.arange(0, inp_deep.shape[1]), inp_deep[index].detach())
                axes2[q, mod].contourf(grid[:, :, 0], grid[:, :, 1], out_deep[index].detach(), cmap="jet", levels=200)
            fig1.savefig(folder + "/Input" + str(i) + "_" + which + "_" + str(noise) + ".png", dpi=200, bbox_inches='tight')
            fig2.savefig(folder + "/DONetOut" + str(i) + "_" + which + "_" + str(noise) + ".png", dpi=200, bbox_inches='tight')
        out_deep_R = torch.mean(out_deep, 0)
        plt.figure()
        plt.contourf(grid[:, :, 0], grid[:, :, 1], out_deep_R.detach(), cmap="jet", levels=200)
        plt.savefig(folder + "/OutR" + str(i) + "_" + which + "_" + str(noise) + ".png", dpi=200, bbox_inches='tight')
        vmax = torch.max(labels_i)
        vmin = torch.min(labels_i)
        err_1 = ((torch.mean(torch.abs(labels_i - pred_test_i_1) ** p) / torch.mean(torch.abs(labels_i) ** p)) ** (1 / p)).item() * 100
        err_2 = ((torch.mean(torch.abs(labels_i - pred_test_i_2) ** p) / torch.mean(torch.abs(labels_i) ** p)) ** (1 / p)).item() * 100
        err_1 = round(err_1, 1)
        err_2 = round(err_2, 1)
        if which != "eit":
            if which == "curve" or which == "style":
                labels_i = labels_i.T
                pred_test_i_1 = pred_test_i_1.T
                pred_test_i_2 = pred_test_i_2.T
            fig, axes = plt.subplots(1, 3, figsize=(30, 8), dpi=200)

            axes[0].grid(True, which="both", ls=":")
            axes[1].grid(True, which="both", ls=":")
            axes[2].grid(True, which="both", ls=":")

            im2 = axes[0].contourf(grid[:, :, 0], grid[:, :, 1], labels_i.detach(), levels=200, cmap="inferno", vmin=vmin, vmax=vmax)
            im3 = axes[1].contourf(grid[:, :, 0], grid[:, :, 1], pred_test_i_1.detach(), levels=200, cmap="inferno", vmin=vmin, vmax=vmax)
            im4 = axes[2].contourf(grid[:, :, 0], grid[:, :, 1], pred_test_i_2.detach(), levels=200, cmap="inferno", vmin=vmin, vmax=vmax)

            axes[0].set(xlabel=r'$x$', ylabel=r'$y$', title="Exact Coefficient " + r'$a(x,y)$')
            axes[1].set(xlabel=r'$x$', ylabel=r'$y$', title="NIO Predicted Coefficient " + r'$a^*(x,y)$')
            axes[2].set(xlabel=r'$x$', ylabel=r'$y$', title="FCNN Predicted Coefficient " + r'$a^*(x,y)$')
            cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
            plt.colorbar(im3, cax=cax, **kw)

            if which == "curve" or which == "style":
                axes[0].invert_yaxis()
                axes[1].invert_yaxis()
                axes[2].invert_yaxis()

            plt.savefig(folder + "/Sample" + str(i) + "_" + which + ".png", dpi=200, bbox_inches='tight')
        else:
            fig, ax = plt.subplots(1, 3, figsize=(30, 8), subplot_kw=dict(projection='polar'))
            ax[0].grid(False)
            ax[1].grid(False)
            ax[2].grid(False)
            ax[0].contourf(grid[:, :, 0], grid[:, :, 1], labels_i.detach(), levels=200, cmap="inferno", vmin=vmin, vmax=vmax)
            ax[0].set_yticklabels([])
            ax[0].set_theta_zero_location("W")

            ax[1].contourf(grid[:, :, 0], grid[:, :, 1], pred_test_i_1.detach(), levels=200, cmap="inferno", vmin=vmin, vmax=vmax)
            ax[1].set_yticklabels([])
            ax[1].set_theta_zero_location("W")

            im3 = ax[2].contourf(grid[:, :, 0], grid[:, :, 1], pred_test_i_2.detach(), levels=200, cmap="inferno", vmin=vmin, vmax=vmax)
            ax[2].set_yticklabels([])
            ax[2].set_theta_zero_location("W")

            ax[0].set(xlabel=r'$x$', ylabel=r'$y$', title="Exact Coefficient " + r'$a(x,y)$')
            ax[1].set(xlabel=r'$x$', ylabel=r'$y$', title="NIO Predicted Coefficient " + r'$a^*(x,y)$')
            ax[2].set(xlabel=r'$x$', ylabel=r'$y$', title="FCNN Predicted Coefficient " + r'$a^*(x,y)$')
            cax, kw = mpl.colorbar.make_axes([a for a in ax.flat])
            plt.colorbar(im3, cax=cax, **kw)

            plt.savefig(folder + "/Sample" + str(i) + "_" + which + ".png", dpi=200, bbox_inches='tight')
