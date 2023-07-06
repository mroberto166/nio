### Neural Inverse Operators for solving PDE Inverse Problems
This repository is the official implementation of the paper [**Neural Inverse Operators for solving PDE Inverse Problems**](https://openreview.net/pdf?id=S4fEjmWg4X)

<br/><br/>

<img src="Images/NIORB.png" width="800" >

<br/><br/>

#### Requirements
The code is based on python 3 (version 3.7) and the packages required can be installed with
```
python3 -m pip install -r requirements.txt
```
#### Source Data
We cover instances of the Poisson, Helmholtz and Radiative Transport equations.
Data can be downloaded from https://zenodo.org/record/7566430 (14GB).
Alternatively, run the script `download_data.py` which downloads all required data into the appropriate folder (it requires 'wget' to be installed on your system).
```
python3 download_data.py
```
The data for the Seismic Imaging problem can be downloaded at: https://openfwi-lanl.github.io/docs/data.html#vel. 
Then, the h5 file required to run the code can be built by running: `GetStyleData.py` and `GetCurveData.py`

#### Models Training
Each of the benchmarks described in tha peper can be trained by running the python scripts `TrainNio.py`.
In order to ba able to run the script, the following arguments have to be added (in the order):
- name of the folder where to save the results
- flag for the problem 
- flag for the model
- number of workers (usually 0, 1, or 2)

The flag for the problem must be one among:
- `sine` for the Caldéron problem with trigonometric coefficients 
- `eit` for the Caldéron problem with Heart&Lungs
- `helm` for the inverse wave scattering
- `rad` for the radiative transfer problem
- `curve` for the seismic imaging with the CurveVel-A dataset
- `style` for the seismic imaging with the CurveVel-A dataset

The flag for the problem must be one among:
- `nio_new` for NIO
- `fcnn` for Fully Convolutional NN
- `don` for DeepONet

For instance:
```
python3 RunNio.py Example helm nio_new 0

```

The models' hyperparameter can be specified in the corresponding python scripts as well.
To train the InversionNet model (the fully convolutional network baseline for Seismic Imaging) please refer to the GitHub page of Deng et Al (https://arxiv.org/pdf/2111.02926.pdf): https://github.com/lanl/OpenFWI

#### Hyperparameters Grid/Random Search
Cross validation for each model can be run with:

```
python3 ModelSelectionNIO.py model which
```

`which` and `model` must be one of the problems and models above.
For examples 
```
python3 ModelSelectionNIO.py nio_new helm
```
For the Seismic Imaging problem, only NIO and DON models can be run.

The hyperparameters of the models in the Table 1 have been obtained in this way.

The best performing configuration can be obtained by visualizing the results with tensorboard:
```
tensorboard --logdir=NameOfTheModelSelectionFolder
```

If a SLURM remote server is available set `sbatch=True` and `cluster="true"` in the script.

#### Pretrained Models
The models trained and used to compute the errors in Table 1 can be downloaded (9GB) by running:
```
python3 download_models.py
```
*Remark*: the compressed folder has to be unzipped!

#### Error Computations
The errors of the best performing models (Table 1) can be computed by running the script `ComputeNoiseErrors.py`.
It will compute the testing error for all the benchmark, for all the models and for different noise levels (Table 6).




