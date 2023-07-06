import itertools
import os
import sys

import numpy as np

np.random.seed(3545)

random = True
cluster = "true"
sbatch = True
GPU = "None"  # GPU="GeForceGTX1080"  # GPU = "GeForceGTX1080Ti"  # GPU = "TeslaV100_SXM2_32GB"

cpus = 1

folder_name = "ModelSelectionOptL2"

network_architecture = {"n_hidden_layers": [2, 4, 8, 12, 20],
                        "neurons": [16, 20, 50, 100, 200, 500],
                        "act_string": ["relu", "tanh", "leaky_relu"],
                        "dropout_rate": [0],
                        "lr": [0.01, 0.1],
                        "retrain": np.random.randint(0, 1000, 10)}

ndic = {**network_architecture}

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
settings = list(itertools.product(*ndic.values()))

i = 0
if random:
    idx = np.random.choice(len(settings), 50, replace=False)
    settings = np.array(settings)[idx].tolist()
for setup in settings:
    # time.sleep(10)
    print(setup)

    folder_path = "\'" + folder_name + "/Setup_" + str(i) + "\'"
    print("###################################")

    net_arch = {
        "n_hidden_layers": int(setup[0]),
        "neurons": int(setup[1]),
        "act_string": setup[2],
        "dropout_rate": float(setup[3]),
        "lr": float(setup[4]),
        "retrain": int(setup[5])
    }

    arguments = list()
    arguments.append(20)
    arguments.append(folder_path)
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        arguments.append("\\\"" + str(net_arch) + "\\\"")

    else:
        arguments.append(str(net_arch).replace("\'", "\""))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            string_to_exec = "sbatch --time=24:00:00 -n " + str(cpus) + " -G 1 --mem-per-cpu=16384 --wrap=\" python3 ConstOpt.py "
        else:
            string_to_exec = "python3 ContOpt.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + str(arg)
        if cluster and sbatch:
            string_to_exec = string_to_exec + " \""
        print(string_to_exec)
        os.system(string_to_exec)
    i = i + 1
