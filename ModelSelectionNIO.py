import itertools
import os
import sys

import numpy as np

np.random.seed(3545)

model = sys.argv[1]
which = sys.argv[2]

random = True
cluster = "false"
sbatch = False
GPU = "None"  # GPU="GeForceGTX1080"  # GPU = "GeForceGTX1080Ti"  # GPU = "TeslaV100_SXM2_32GB"
ablation_fno = True
cpus = 0
max_workers = int(cpus * 2)
script = "RunNio.py"
if cpus == 0:
    cpus = 1

if which == "sine"  or which == "helm":
    if model == "nio":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.001],
            # "norm": ["none", "norm", "minmax", "norm-inp"],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [2],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [50],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [4, 6, 8],
            "neurons_t": [100, 200],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [25, 100, 400]
        }

        fno_architecture_ = {
            "width": [32, 64],
            "modes": [16, 25, 50],
            "n_layers": [2, 3, 4]
        }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [50],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
    if model == "fcnn":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.0005, 0.001],
            # "norm": ["none", "minmax", "norm-inp", "norm"],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [2],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [16, 32, 64, 128],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [4],
            "neurons_t": [50],
            "act_string_t": ["tanh"],
            "dropout_rate_t": [0.0],
            "n_basis": [25]
        }

        fno_architecture_ = {
            "width": [32],
            "modes": [4],
            "n_layers": [2]
        }
        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [50],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
    if model == "don":
        # DeepONet
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.001],
            # "norm": ["none", "norm", "minmax", "norm-inp"],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [1],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [50],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [6, 8, 12, 15],
            "neurons_t": [100, 200, 500],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [25, 100, 400, 1000]
        }

        fno_architecture_ = {
            "width": [32],
            "modes": [16],
            "n_layers": [0]
        }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [50],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
    if model == "fcnio":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.001],
            # "norm": ["none", "norm", "minmax", "norm-inp"],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [2],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [4, 6, 8],
            "neurons_b": [100, 200],
            "act_string_b": ["leaky_relu"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [4, 6, 8],
            "neurons_t": [100, 200],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [25, 100, 400]
        }

        fno_architecture_ = {
            "width": [32, 64],
            "modes": [16, 25, 50],
            "n_layers": [2, 3, 4]
        }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [4, 6, 8],
            "neurons_db": [1000],
            "act_string_db": ["leaky_relu"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
    if model == "nio_new":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.001],
            # "norm": ["none", "norm", "minmax", "norm-inp"],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [2],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [50],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [4, 6, 8],
            "neurons_t": [100, 200],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [25, 100, 400]
        }
        if not ablation_fno:
            fno_architecture_ = {
                "width": [32, 64],
                "modes": [16, 25],
                "n_layers": [2, 3, 4]
            }
        else:
            fno_architecture_ = {
                "width": [1],
                "modes": [0],
                "n_layers": [0]
            }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [50],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
if which == "eit":

    if model == "nio":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.001],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [2],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [50],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [4, 6, 8],
            "neurons_t": [100, 200],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [25, 100, 400]
        }

        fno_architecture_ = {
            "width": [32, 64],
            "modes": [16, 25, 50],
            "n_layers": [2, 3, 4]
        }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [50],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
    if model == "fcnn":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.0005, 0.001],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [2],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [16, 32, 64, 128],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [4],
            "neurons_t": [50],
            "act_string_t": ["tanh"],
            "dropout_rate_t": [0.0],
            "n_basis": [25]
        }

        fno_architecture_ = {
            "width": [32],
            "modes": [4],
            "n_layers": [2]
        }
        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [50],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
    if model == "don":
        # DeepONet
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.001],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [1],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [50],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [6, 8, 12, 15],
            "neurons_t": [100, 200, 500],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [25, 100, 400, 1000]
        }

        fno_architecture_ = {
            "width": [32],
            "modes": [16],
            "n_layers": [0]
        }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [50],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
    if model == "nio_new":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.001],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [2],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [50],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [4, 6, 8],
            "neurons_t": [100, 200],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [25, 100, 400]
        }

        if not ablation_fno:
            fno_architecture_ = {
                "width": [32, 64],
                "modes": [16, 25, 50],
                "n_layers": [2, 3, 4]
            }
        else:
            fno_architecture_ = {
                "width": [1],
                "modes": [0],
                "n_layers": [0]
            }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [1000],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
if which == "rad":

    if model == "nio":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.001],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [2],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [50],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [4, 6, 8],
            "neurons_t": [100, 200],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [25, 100, 400]
        }

        fno_architecture_ = {
            "width": [32, 64, 128],
            "modes": [16, 32],
            "n_layers": [3, 4]
        }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [50],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
    if model == "fcnn":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.0005, 0.001],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [2],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [16, 32, 64],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [4],
            "neurons_t": [50],
            "act_string_t": ["tanh"],
            "dropout_rate_t": [0.0],
            "n_basis": [25]
        }

        fno_architecture_ = {
            "width": [32],
            "modes": [4],
            "n_layers": [2]
        }
        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [50],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
    if model == "don":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.001],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [1],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [50],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [6, 8, 12, 15],
            "neurons_t": [100, 200, 500],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [25, 100, 400, 1000]
        }

        fno_architecture_ = {
            "width": [32],
            "modes": [16],
            "n_layers": [0]
        }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [50],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
    if model == "nio_new":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.001],
            "norm": ["none", "minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [2],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [50],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [4, 6, 8],
            "neurons_t": [100, 200],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [25, 100, 400]
        }

        fno_architecture_ = {
            "width": [32, 64],
            "modes": [16, 32],
            "n_layers": [2, 3, 4]
        }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [1000],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
if which == "style" or which == "curve":

    if model == "nio":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.97],
            "epochs": [120],
            "batch_size": [256],
            "learning_rate": [0.001],
            "norm": ["none", "minmax", "log-minmax"],
            "weight_decay": [0.],
            "reg_param": [0.],
            "reg_exponent": [2],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [6],
            "neurons_b": [128],
            "act_string_b": ["leaky_relu"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [8],
            "neurons_t": [256],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [50, 100, 500]
        }

        fno_architecture_ = {
            "width": [32, 64],
            "modes": [16, 24],
            "n_layers": [2, 3]
        }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [3],
            "neurons_db": [1000],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
    if model == "don":
        # DeepONet
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [120],
            "batch_size": [256],
            "learning_rate": [0.001],
            "norm": ["none", "minmax", "log-minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [1],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [50],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [6, 8, 12, 15],
            "neurons_t": [100, 200, 500],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [25, 100, 400, 1000]
        }

        fno_architecture_ = {
            "width": [32],
            "modes": [16],
            "n_layers": [0]
        }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [50],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }
    if model == "nio_new":
        training_properties_ = {
            "step_size": [15],
            "gamma": [1, 0.98],
            "epochs": [1000],
            "batch_size": [256],
            "learning_rate": [0.001],
            "norm": ["none", "minmax", "log-minmax"],
            "weight_decay": [0., 1e-6],
            "reg_param": [0.],
            "reg_exponent": [2],
            "inputs": [2],
            "b_scale": [0.],
            "retrain": np.concatenate([(4 * np.ones(1, )).astype(int), np.random.randint(0, 1000, 1)]),
            "mapping_size_ff": [32],
            "scheduler": ["step"]
        }
        branch_architecture_ = {
            "n_hidden_layers_b": [2],
            "neurons_b": [50],
            "act_string_b": ["tanh"],
            "dropout_rate_b": [0.0],
            "kernel_size": [3],
        }

        trunk_architecture_ = {
            "n_hidden_layers_t": [4, 6, 8],
            "neurons_t": [100, 200],
            "act_string_t": ["leaky_relu"],
            "dropout_rate_t": [0.0],
            "n_basis": [25, 100, 400]
        }

        fno_architecture_ = {
            "width": [32, 64],
            "modes": [16, 25],
            "n_layers": [2, 3, 4]
        }

        denseblock_architecture_ = {
            "n_hidden_layers_db": [2],
            "neurons_db": [1000],
            "act_string_db": ["tanh"],
            "retrain_db": [127],
            "dropout_rate_db": [0.0],
        }

if ablation_fno:

    folder_name = "ModelSelection_final_s_abl_" + which + "_" + model
else:
    folder_name = "ModelSelection_final_s_" + which + "_" + model

ndic = {**training_properties_,
        **branch_architecture_,
        **trunk_architecture_,
        **fno_architecture_,
        **denseblock_architecture_}

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
settings = list(itertools.product(*ndic.values()))

i = 0
if model == "nio_new":
    nconf = 30
else:
    nconf = 30

if len(settings) < nconf:
    nconf = len(settings)

if random:
    idx = np.random.choice(len(settings), nconf, replace=False)
    settings = np.array(settings)[idx].tolist()

for setup in settings:
    # time.sleep(10)
    print(setup)

    folder_path = "\'" + folder_name + "/Setup_" + str(i) + "\'"
    print("###################################")
    training_properties_ = {
        "step_size": int(setup[0]),
        "gamma": float(setup[1]),
        "epochs": int(setup[2]),
        "batch_size": int(setup[3]),
        "learning_rate": float(setup[4]),
        "norm": setup[5],
        "weight_decay": float(setup[6]),
        "reg_param": float(setup[7]),
        "reg_exponent": int(setup[8]),
        "inputs": int(setup[9]),
        "b_scale": float(setup[10]),
        "retrain": int(setup[11]),
        "mapping_size_ff": int(setup[12]),
        "scheduler": setup[13]
    }

    branch_architecture_ = {
        "n_hidden_layers": int(setup[14]),
        "neurons": int(setup[15]),
        "act_string": setup[16],
        "dropout_rate": float(setup[17]),
        "kernel_size": int(setup[18])
    }

    trunk_architecture_ = {
        "n_hidden_layers": int(setup[19]),
        "neurons": int(setup[20]),
        "act_string": setup[21],
        "dropout_rate": float(setup[22]),
        "n_basis": int(setup[23])
    }

    fno_architecture_ = {
        "width": int(setup[24]),
        "modes": int(setup[25]),
        "n_layers": int(setup[26])
    }

    denseblock_architecture_ = {
        "n_hidden_layers": int(setup[27]),
        "neurons": int(setup[28]),
        "act_string": setup[29],
        "retrain": int(setup[30]),
        "dropout_rate": float(setup[31])
    }
    arguments = list()
    arguments.append(folder_path)
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if sbatch:
            arguments.append("\\\"" + str(training_properties_) + "\\\"")
        else:
            arguments.append("\'" + str(training_properties_).replace("\'", "\"") + "\'")

    else:
        arguments.append(str(training_properties_).replace("\'", "\""))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if sbatch:
            arguments.append("\\\"" + str(branch_architecture_) + "\\\"")
        else:
            arguments.append("\'" + str(branch_architecture_).replace("\'", "\"") + "\'")

    else:
        arguments.append(str(branch_architecture_).replace("\'", "\""))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        #
        if sbatch:
            arguments.append("\\\"" + str(trunk_architecture_) + "\\\"")
        else:
            arguments.append("\'" + str(trunk_architecture_).replace("\'", "\"") + "\'")
    else:
        arguments.append(str(trunk_architecture_).replace("\'", "\""))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        # arguments.append("\'" + str(fno_architecture_).replace("\'", "\"") + "\'")
        if sbatch:
            arguments.append("\\\"" + str(fno_architecture_) + "\\\"")
        else:
            arguments.append("\'" + str(fno_architecture_).replace("\'", "\"") + "\'")
    else:
        arguments.append(str(fno_architecture_).replace("\'", "\""))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if sbatch:
            arguments.append("\\\"" + str(denseblock_architecture_) + "\\\"")
        else:
            arguments.append("\'" + str(denseblock_architecture_).replace("\'", "\"") + "\'")

    else:
        arguments.append(str(denseblock_architecture_).replace("\'", "\""))

    arguments.append(which)
    arguments.append(model)
    arguments.append(max_workers)
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            # string_to_exec = "bsub -W 16:00 -n 32 -R \'rusage[mem=2048]\' -R \'rusage[ngpus_excl_p=1]\' python3 OpSquareBigData.py"
            # string_to_exec = "bsub -W 16:00 -n 32 -R \'rusage[mem=2048]\' -R \'rusage[ngpus_excl_p=1]\' python3 CNOFWI.py"
            # string_to_exec = "bsub -W 16:00 -n 32 -R \'rusage[mem=2048]\' -R \'rusage[ngpus_excl_p=1]\' python3 OpSquarePoissonNew.py"
            # string_to_exec = "bsub -W 16:00 -n 32 -R \'rusage[mem=2048]\' -R \'rusage[ngpus_excl_p=1]\' python3 OpSquareRad.py"
            if sbatch:
                if which == "curve" or which == "style":
                    string_to_exec = "sbatch --time=72:00:00 -n " + str(cpus) + " -G 1 --mem-per-cpu=16384 --wrap=\" python3 " + script + " "
                else:
                    string_to_exec = "sbatch --time=24:00:00 -n " + str(cpus) + " -G 1 --mem-per-cpu=16384 --wrap=\" python3 " + script + " "
        else:
            string_to_exec = "python3 " + script + " "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + str(arg)
        if cluster and sbatch:
            string_to_exec = string_to_exec + " \""
        print(string_to_exec)
        os.system(string_to_exec)
    i = i + 1
