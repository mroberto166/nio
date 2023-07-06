import copy
import json
import os
import random
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from GPUtil.GPUtil import getGPUs
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Baselines import InversionNetHelm, InversionNetRad, InversionNetEIT
from NIOModules import NIOHelmPermInv, NIOHeartPerm, NIORadPerm, NIOWavePerm
from NIOModules import SNOHelmConv, SNOWaveConv2, SNOConvEIT, SNOConvRad
from debug_tools import CudaMemoryDebugger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

seed = 0
random.seed(seed)  # python random generator
np.random.seed(seed)  # numpy random generator

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

folder = sys.argv[1]

freq_print = 1

if len(sys.argv) == 5:
    training_properties_ = {
        "step_size": 15,
        "gamma": 1,
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.001,
        "norm": "log-minmax",
        "weight_decay": 0,
        "reg_param": 0,
        "reg_exponent": 1,
        "inputs": 2,
        "b_scale": 0.,
        "retrain": 888,
        "mapping_size_ff": 32,
        "scheduler": "step"
    }

    branch_architecture_ = {
        "n_hidden_layers": 3,
        "neurons": 64,
        "act_string": "leaky_relu",
        "dropout_rate": 0.0,
        "kernel_size": 3
    }

    trunk_architecture_ = {
        "n_hidden_layers": 8,
        "neurons": 256,
        "act_string": "leaky_relu",
        "dropout_rate": 0.0,
        "n_basis": 50
    }

    fno_architecture_ = {
        "width": 64,
        "modes": 16,
        "n_layers": 1,
    }

    denseblock_architecture_ = {
        "n_hidden_layers": 4,
        "neurons": 2000,
        "act_string": "leaky_relu",
        "retrain": 56,
        "dropout_rate": 0.0
    }

    problem = sys.argv[2]
    mod = sys.argv[3]
    max_workers = int(sys.argv[4])
else:
    training_properties_ = json.loads(sys.argv[2].replace("\'", "\""))

    branch_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))

    trunk_architecture_ = json.loads(sys.argv[4].replace("\'", "\""))

    fno_architecture_ = json.loads(sys.argv[5].replace("\'", "\""))

    denseblock_architecture_ = json.loads(sys.argv[6].replace("\'", "\""))

    problem = sys.argv[7]
    mod = sys.argv[8]
    max_workers = int(sys.argv[9])

if problem == "sine":
    from Problems.PoissonSin import MyDataset

    padding_frac = 1 / 4
elif problem == "helm":
    from Problems.HelmNIO import MyDataset

    padding_frac = 1 / 4

elif problem == "curve":
    from Problems.CurveVel import MyDataset

    padding_frac = 1 / 4

elif problem == "style":
    from Problems.StlyleData import MyDataset

    padding_frac = 1 / 4

elif problem == "rad":
    from Problems.AlbedoOperator import MyDataset

    padding_frac = 1 / 4

elif problem == "eit":
    from Problems.HeartLungsEIT import MyDataset

    padding_frac = 1 / 4
if torch.cuda.is_available():
    memory_avail = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print("Running on ", torch.cuda.get_device_name(0), "Total memory: ", memory_avail, " GB")

print_mem = False
disable = False
full_training = False

step_size = training_properties_["step_size"]
gamma = training_properties_["gamma"]
norm = training_properties_["norm"]
epochs = training_properties_["epochs"]
batch_size = training_properties_["batch_size"]
learning_rate = training_properties_["learning_rate"]
weight_decay = training_properties_["weight_decay"]
reg_param = training_properties_["reg_param"]
reg_exponent = training_properties_["reg_exponent"]
inputs_bool = training_properties_["inputs"]
retrain_seed = training_properties_["retrain"]
b_scale = training_properties_["b_scale"]
mapping_size = training_properties_["mapping_size_ff"]
scheduler_string = training_properties_["scheduler"]

dict_hp = training_properties_.copy()
branch_architecture_copy = branch_architecture_.copy()

branch_architecture_copy["n_hidden_layers_b"] = branch_architecture_copy.pop("n_hidden_layers")
branch_architecture_copy["dropout_rate_b"] = branch_architecture_copy.pop("dropout_rate")
branch_architecture_copy["neurons_b"] = branch_architecture_copy.pop("neurons")
branch_architecture_copy["act_string_b"] = branch_architecture_copy.pop("act_string")

dict_hp.update(branch_architecture_copy)
dict_hp.update(trunk_architecture_)
dict_hp.update(fno_architecture_)

fno_input_dimension = denseblock_architecture_["neurons"]
cuda_debugger = CudaMemoryDebugger(print_mem)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = MyDataset(norm=norm, inputs_bool=inputs_bool, device=device, which="training", mod=mod)
test_dataset = MyDataset(norm=norm, inputs_bool=inputs_bool, device=device, which="validation", mod=mod, noise=0.1)
inp_dim_branch = train_dataset.inp_dim_branch
n_fun_samples = train_dataset.n_fun_samples

grid = train_dataset.get_grid().squeeze(0)

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

    df = pd.DataFrame.from_dict([training_properties_]).T
    df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='a')

    df = pd.DataFrame.from_dict([branch_architecture_]).T
    df.to_csv(folder + '/branch_architecture.txt', header=False, index=True, mode='a')

    df = pd.DataFrame.from_dict([trunk_architecture_]).T
    df.to_csv(folder + '/trunk_architecture.txt', header=False, index=True, mode='a')

    df = pd.DataFrame.from_dict([fno_architecture_]).T
    df.to_csv(folder + '/fno_architecture.txt', header=False, index=True, mode='a')

    df = pd.DataFrame.from_dict([denseblock_architecture_]).T
    df.to_csv(folder + '/denseblock_architecture.txt', header=False, index=True, mode='a')
    if mod == "nio" or mod == "don":
        print("Using CNIO")
        if problem == "sine" or problem == "helm" or problem == "step":
            model = SNOHelmConv(input_dimensions_branch=inp_dim_branch,
                                input_dimensions_trunk=grid.shape[2],
                                network_properties_branch=branch_architecture_,
                                network_properties_trunk=trunk_architecture_,
                                fno_architecture=fno_architecture_,
                                device=device,
                                retrain_seed=retrain_seed)
        elif problem == "curve" or problem == "style":
            model = SNOWaveConv2(input_dimensions_branch=inp_dim_branch,
                                 input_dimensions_trunk=grid.shape[2],
                                 network_properties_branch=branch_architecture_,
                                 network_properties_trunk=trunk_architecture_,
                                 fno_architecture=fno_architecture_,
                                 device=device,
                                 retrain_seed=retrain_seed,
                                 b_scale=b_scale,
                                 mapping_size=mapping_size)
        elif problem == "rad":
            model = SNOConvRad(input_dimensions_branch=inp_dim_branch,
                               input_dimensions_trunk=1,
                               network_properties_branch=branch_architecture_,
                               network_properties_trunk=trunk_architecture_,
                               fno_architecture=fno_architecture_,
                               device=device,
                               retrain_seed=retrain_seed)
        elif problem == "eit":
            model = SNOConvEIT(input_dimensions_branch=inp_dim_branch,
                               input_dimensions_trunk=grid.shape[2],
                               network_properties_branch=branch_architecture_,
                               network_properties_trunk=trunk_architecture_,
                               fno_architecture=fno_architecture_,
                               device=device,
                               retrain_seed=retrain_seed)
    elif mod == "fcnn":
        print("Using FCNN")
        if problem == "sine" or problem == "helm" or problem == "step":
            model = InversionNetHelm(int(branch_architecture_["neurons"]))
        elif problem == "rad":
            model = InversionNetRad(int(branch_architecture_["neurons"]))
        elif problem == "eit":
            model = InversionNetEIT(int(branch_architecture_["neurons"]))
    else:
        print("Using FCNIO")
        if problem == "sine" or problem == "helm" or problem == "step":
            model = NIOHelmPermInv(input_dimensions_branch=inp_dim_branch,
                                   input_dimensions_trunk=grid.shape[2],
                                   network_properties_branch=branch_architecture_,
                                   network_properties_trunk=trunk_architecture_,
                                   fno_architecture=fno_architecture_,
                                   device=device,
                                   retrain_seed=retrain_seed,
                                   fno_input_dimension=fno_input_dimension)

        elif problem == "curve" or problem == "style":
            model = NIOWavePerm(input_dimensions_branch=inp_dim_branch,
                                input_dimensions_trunk=grid.shape[2],
                                network_properties_branch=branch_architecture_,
                                network_properties_trunk=trunk_architecture_,
                                fno_architecture=fno_architecture_,
                                device=device,
                                retrain_seed=retrain_seed,
                                fno_input_dimension=fno_input_dimension)
        elif problem == "rad":
            model = NIORadPerm(input_dimensions_branch=inp_dim_branch,
                               input_dimensions_trunk=1,
                               network_properties_branch=branch_architecture_,
                               network_properties_trunk=trunk_architecture_,
                               fno_architecture=fno_architecture_,
                               device=device,
                               retrain_seed=retrain_seed,
                               fno_input_dimension=fno_input_dimension)
        elif problem == "eit":
            model = NIOHeartPerm(input_dimensions_branch=inp_dim_branch,
                                 input_dimensions_trunk=2,
                                 network_properties_branch=branch_architecture_,
                                 network_properties_trunk=trunk_architecture_,
                                 fno_architecture=fno_architecture_,
                                 device=device,
                                 retrain_seed=retrain_seed,
                                 fno_input_dimension=fno_input_dimension)
    start_epoch = 0
    best_model_testing_error = 100
    best_model = None
    print("Using", torch.cuda.device_count(), "GPUs!")

else:
    print("Folder already exists! Looking for the model")

    if os.path.isfile(folder + "/model.pkl"):
        print("Found and loading existing model")
        model = torch.load(folder + "/model.pkl")
        errors = pd.read_csv(folder + "/errors.txt", header=None, sep=":", index_col=0)
        errors = errors.transpose().reset_index().drop("index", 1)
        start_epoch = int(errors["Current Epoch"].values[0]) + 1
        best_model_testing_error = float(errors["Best Testing Error"].values[0])
        best_model = copy.deepcopy(model)

    else:
        print("Found no model. Creating a new one")
        if mod == "nio" or mod == "don":
            if problem == "sine" or problem == "helm" or problem == "step":
                model = SNOHelmConv(input_dimensions_branch=inp_dim_branch,
                                    input_dimensions_trunk=grid.shape[2],
                                    network_properties_branch=branch_architecture_,
                                    network_properties_trunk=trunk_architecture_,
                                    fno_architecture=fno_architecture_,
                                    device=device,
                                    retrain_seed=retrain_seed)
            elif problem == "curve" or problem == "style":
                model = SNOWaveConv2(input_dimensions_branch=inp_dim_branch,
                                     input_dimensions_trunk=grid.shape[2],
                                     network_properties_branch=branch_architecture_,
                                     network_properties_trunk=trunk_architecture_,
                                     fno_architecture=fno_architecture_,
                                     device=device,
                                     retrain_seed=retrain_seed,
                                     b_scale=b_scale,
                                     mapping_size=mapping_size)
            elif problem == "rad":
                model = SNOConvRad(input_dimensions_branch=inp_dim_branch,
                                   input_dimensions_trunk=1,
                                   network_properties_branch=branch_architecture_,
                                   network_properties_trunk=trunk_architecture_,
                                   fno_architecture=fno_architecture_,
                                   device=device,
                                   retrain_seed=retrain_seed)
            elif problem == "eit":
                model = SNOConvEIT(input_dimensions_branch=inp_dim_branch,
                                   input_dimensions_trunk=grid.shape[2],
                                   network_properties_branch=branch_architecture_,
                                   network_properties_trunk=trunk_architecture_,
                                   fno_architecture=fno_architecture_,
                                   device=device,
                                   retrain_seed=retrain_seed)
        elif mod == "fcnn":
            if problem == "sine" or problem == "helm" or problem == "step":
                model = InversionNetHelm(int(branch_architecture_["neurons"]))
            elif problem == "rad":
                model = InversionNetRad(int(branch_architecture_["neurons"]))
            elif problem == "eit":
                model = InversionNetEIT(int(branch_architecture_["neurons"]))
        else:
            if problem == "sine" or problem == "helm" or problem == "step":
                model = NIOHelmPermInv(input_dimensions_branch=inp_dim_branch,
                                       input_dimensions_trunk=grid.shape[2],
                                       network_properties_branch=branch_architecture_,
                                       network_properties_trunk=trunk_architecture_,
                                       fno_architecture=fno_architecture_,
                                       device=device,
                                       retrain_seed=retrain_seed,
                                       fno_input_dimension=fno_input_dimension)
            elif problem == "curve" or problem == "style":
                model = NIOWavePerm(input_dimensions_branch=inp_dim_branch,
                                    input_dimensions_trunk=grid.shape[2],
                                    network_properties_branch=branch_architecture_,
                                    network_properties_trunk=trunk_architecture_,
                                    fno_architecture=fno_architecture_,
                                    device=device,
                                    retrain_seed=retrain_seed,
                                    fno_input_dimension=fno_input_dimension)
            elif problem == "rad":
                model = NIORadPerm(input_dimensions_branch=inp_dim_branch,
                                   input_dimensions_trunk=1,
                                   network_properties_branch=branch_architecture_,
                                   network_properties_trunk=trunk_architecture_,
                                   fno_architecture=fno_architecture_,
                                   device=device,
                                   retrain_seed=retrain_seed,
                                   fno_input_dimension=fno_input_dimension)

            elif problem == "eit":
                model = NIOHeartPerm(input_dimensions_branch=inp_dim_branch,
                                     input_dimensions_trunk=2,
                                     network_properties_branch=branch_architecture_,
                                     network_properties_trunk=trunk_architecture_,
                                     fno_architecture=fno_architecture_,
                                     device=device,
                                     retrain_seed=retrain_seed,
                                     fno_input_dimension=fno_input_dimension)

        start_epoch = 0
        best_model_testing_error = 100
        best_model = None

model.to(device)
size = model.print_size()
f = open(folder + "/size.txt", "w")
print(size, file=f)

batch_acc = 16
if torch.cuda.is_available():
    batch_acc = batch_acc * torch.cuda.device_count()

print("Maximum number of workers: ", max_workers)
training_set = DataLoader(train_dataset, batch_size=batch_acc, shuffle=True, num_workers=max_workers, pin_memory=True)
testing_set = DataLoader(test_dataset, batch_size=40, shuffle=True, num_workers=max_workers, pin_memory=True)
n_iter_per_epoch = int((train_dataset.length + 1) / batch_size)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

if scheduler_string == "cyclic":
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, cycle_momentum=False, step_size_up=int(n_iter_per_epoch / 2) * epochs, mode="triangular2")
elif scheduler_string == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_iter_per_epoch, gamma=gamma)
else:
    raise ValueError
if os.path.isfile(folder + "/optimizer_state.pkl"):
    checkpoint = torch.load(folder + "/optimizer_state.pkl")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
p = 1
if p == 2:
    my_loss = torch.nn.MSELoss()
elif p == 1:
    my_loss = torch.nn.L1Loss()
else:
    raise ValueError("Choose p = 1 or p=2")

loss_eval = torch.nn.L1Loss()
writer = SummaryWriter(log_dir=folder)

cuda_debugger.print("Beginning")
lr_all = list()
training_all = list()

counter = 0
patience = int(0.1 * epochs)
time_per_epoch = 0
for epoch in range(start_epoch, epochs + start_epoch):
    bar = tqdm(unit="batch", disable=disable)
    with bar as tepoch:
        start = timer()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        running_relative_train_mse = 0.0
        model.train()
        grid = grid.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        for step, (input_batch, output_batch) in enumerate(training_set):

            if torch.cuda.is_available():
                mem = str(round(getGPUs()[0].memoryUtil, 2) * 100) + "%"
            else:
                mem = str(0.) + "%"

            tepoch.update(1)
            input_batch = input_batch.to(device, non_blocking=True)
            output_batch = output_batch.to(device, non_blocking=True)
            cuda_debugger.print("Loading")

            pred_train = model(input_batch, grid)

            cuda_debugger.print("Forward")

            loss_f = my_loss(pred_train, output_batch) / torch.mean(abs(output_batch) ** p) ** (1 / p)
            if reg_param != 0:
                loss_f += reg_param * model.regularization(reg_exponent)
            cuda_debugger.print("Loss Computation")
            loss_f.backward()
            cuda_debugger.print("Backward")

            ########################################################################################
            # Evaluation
            ########################################################################################
            train_mse = train_mse * step / (step + 1) + loss_f / (step + 1)
            tepoch.set_postfix({'Batch': step + 1,
                                'Train loss (in progress)': train_mse.item(),
                                'lr': scheduler.get_last_lr()[0],
                                "GPU Mem": mem,
                                "Patience:": counter,
                                })
            if (step + 1) % int(batch_size / batch_acc) == 0 or (step + 1) == len(training_set):
                optimizer.step()  # Now we can do an optimizer step
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if p == 1:
            writer.add_scalar("train_loss/L1 Error", train_mse, epoch)
        if p == 2:
            writer.add_scalar("train_loss/L2 Error", train_mse, epoch)
        end = timer()
        elapsed = end - start
        ########################################################################################
        # Evaluation
        ########################################################################################
        if epoch % freq_print == 0:
            if not full_training:
                running_relative_test_mse = 0.0

                model.eval()
                with torch.no_grad():
                    for step, (input_batch, output_batch) in enumerate(testing_set):
                        input_batch = input_batch.to(device, non_blocking=True)
                        output_batch = output_batch.to(device, non_blocking=True)
                        pred_test = model(input_batch, grid)
                        pred_test = train_dataset.denormalize(pred_test)
                        output_batch = train_dataset.denormalize(output_batch)
                        loss_test = loss_eval(pred_test, output_batch) / loss_eval(torch.zeros_like(output_batch).to(device), output_batch)
                        running_relative_test_mse = running_relative_test_mse * step / (step + 1) + loss_test.item() ** (1 / p) * 100 / (step + 1)

                    writer.add_scalar("val_loss/Relative Testing Error", running_relative_test_mse, epoch)

            else:
                running_relative_test_mse = train_mse.item()

            if running_relative_test_mse < best_model_testing_error:
                best_model_testing_error = running_relative_test_mse
                torch.save(model, folder + "/model.pkl")
                writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
                writer.add_scalar("time/Elapsed", elapsed, epoch)
                counter = 0
            else:
                counter += 1
        else:
            torch.save(model, folder + "/model.pkl")

        time_per_epoch = time_per_epoch * epoch / (epoch + 1) + elapsed / (epoch + 1)

        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(running_relative_train_mse) + "\n")
            file.write("Testing Error: " + str(running_relative_test_mse) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Time per Epoch: " + str(time_per_epoch) + "\n")
            file.write("Workers: " + str(max_workers) + "\n")

        torch.save({'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, folder + "/optimizer_state.pkl")
        tepoch.set_postfix({"Val loss": running_relative_test_mse})
        tepoch.close()

    if counter > patience:
        print("Early stopping:", epoch, counter)
        break

writer.flush()
writer.close()
