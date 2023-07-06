import h5py
import matplotlib.pyplot as plt
import numpy as np

base_path = "dataStyle/"

hf = h5py.File('dataStyle/StyleData.h5', 'w')
hf.create_group("training")
hf.create_group("validation")
hf.create_group("testing")
mean_fun_inp = 0.
mean_fun_out = 0.

std_fun_inp = 0.
std_fun_out = 0.
k = 1

training_size = 55000
val_size = 5000
tot = training_size + val_size + 7000
for j in range(1, 135):
    model_path = base_path + "model/model" + str(j) + ".npy"
    data_path = base_path + "data/data" + str(j) + ".npy"
    model = np.load(model_path)
    data = np.load(data_path)
    for i in range(data.shape[0]):

        name = "sample_" + str(k - 1)
        if k < training_size:
            which = "training"
            old_mean_inp = mean_fun_inp
            old_mean_out = mean_fun_out

            mean_fun_inp = mean_fun_inp * (k - 1) / k + data[i] / k
            std_fun_inp = std_fun_inp + ((data[i] - mean_fun_inp) * (data[i] - old_mean_inp) - std_fun_inp) / k

            mean_fun_out = mean_fun_out * (k - 1) / k + model[i, 0] / k
            std_fun_out = std_fun_out + ((model[i, 0] - mean_fun_out) * (model[i, 0] - old_mean_out) - std_fun_out) / k

        if training_size <= k < training_size + val_size:
            which = "validation"
        if k >= training_size + val_size:
            which = "testing"
        print(which, k)
        hf[which].create_group(name)
        hf[which][name].create_dataset("input", data=data[i])
        hf[which][name].create_dataset("output", data=model[i, 0])
        k = k + 1

print(std_fun_inp[std_fun_inp < 0])
std_fun_inp = std_fun_inp ** 0.5

print(std_fun_out[std_fun_out < 0])
std_fun_out = std_fun_out ** 0.5

hf.create_dataset("mean_inp_fun", data=mean_fun_inp)
hf.create_dataset("mean_out_fun", data=mean_fun_out)
hf.create_dataset("std_inp_fun", data=std_fun_inp)
hf.create_dataset("std_out_fun", data=std_fun_out)

plt.figure()
plt.imshow(mean_fun_inp[2], aspect="auto")
plt.figure()
plt.imshow(mean_fun_out)

plt.figure()
plt.imshow(std_fun_inp[2], aspect="auto")

plt.figure()
plt.imshow(std_fun_out)
plt.show()
