import ast
import itertools
import os
import sys

import numpy as np

np.random.seed(3545)

random = True
cluster = "true"
sbatch = True
cpus = 0
max_workers = int(cpus * 2)
model = "nio_new"
if cpus == 0:
    cpus = 1
for which in ["eit", "rad"]:
    for abl_fno in [True, False]:
        print("------------------------------------------------------------------------------------------------")
        training_properties_ = dict()
        branch_architecture_ = dict()
        trunk_architecture_ = dict()
        fno_architecture_ = dict()
        denseblock_architecture_ = dict()

        dictionaries = [training_properties_, branch_architecture_, trunk_architecture_, fno_architecture_, denseblock_architecture_]
        paths = ["/training_properties.txt", "/branch_architecture.txt", "/trunk_architecture.txt", "/fno_architecture.txt", "/denseblock_architecture.txt"]
        path = "FinalModelNewPerm/Best_" + model + "_" + which
        for dictionary, p in zip(dictionaries, paths):
            with open(path + p) as f:
                for line in f:
                    (key, val) = line.replace("\n", "").split(",")
                    if "branch" in p:
                        key = key + "_b"
                    if "trunk" in p:
                        key = key + "_t"
                    if "dense" in p:
                        key = key + "_db"
                    if "norm" not in key and "act" not in key and "scheduler" not in key and "retrain" not in key:
                        dictionary[key] = list([ast.literal_eval(val)])
                    elif "retrain" in key:
                        val = ast.literal_eval(val)
                        dictionary[key] = list([val])
                    else:
                        dictionary[key] = list([val])

        if abl_fno:
            fno_architecture_ = {
                "width": [1],
                "modes": [0],
                "n_layers": [0]
            }

        # errors = pd.read_csv(path + "/errors.txt", header=None, sep=":", index_col=0)
        # errors = errors.transpose().reset_index().drop("index", 1)
        # epoch = errors["Current Epoch"]
        # training_properties_["epochs"] = [int(epoch)]

        if abl_fno:
            script = "RunNio.py"
            folder_name = "ModelSelection_final_abl_fno_conv_" + which + "_" + model
        else:
            script = "RunNioAbl.py"
            folder_name = "ModelSelection_final_abl_rb_" + which + "_" + model

        print(training_properties_)
        print(branch_architecture_)
        print(trunk_architecture_)
        print(fno_architecture_)
        print(denseblock_architecture_)
        print(script, folder_name, abl_fno)
        ndic = {**training_properties_,
                **branch_architecture_,
                **trunk_architecture_,
                **fno_architecture_,
                **denseblock_architecture_}

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        settings = list(itertools.product(*ndic.values()))

        i = 0
        nconf = 1

        if len(settings) < nconf:
            nconf = len(settings)

        if random:
            idx = np.random.choice(len(settings), nconf, replace=False)
            settings = np.array(settings)[idx].tolist()

        for setup in settings:
            # time.sleep(10)

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
            arguments.append("nio_new")
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
                # print(string_to_exec)
                os.system(string_to_exec)
            i = i + 1
