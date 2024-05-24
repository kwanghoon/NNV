# NNV Environment Setup and Training Guide

## Prerequisites

1. Download and install Anaconda from:
   - [Anaconda Official Website](https://www.anaconda.com/)
   - [Benn's Blog (Korean)](https://benn.tistory.com/26)

## Setup Instructions

### Step 1: Open Anaconda PowerShell Prompt

- Open the Start Menu.
- Search for "Anaconda Powershell Prompt" and open it.

### Step 2: Create and Activate Virtual Environment

```shell
D:\> conda create -n nnv
D:\> conda activate nnv
D:\> conda install pip
```

### Step 3: Install Required Packages

```shell
D:\> pip install numpy
D:\> pip install scipy
D:\> pip install matplotlib
D:\> pip install tensorflow
D:\> pip install scikit-learn
```

### Step 4: Create Necessary Directories

```shell
D:\> mkdir History
D:\> mkdir Logs
D:\> mkdir Models
D:\> mkdir Prediction
D:\> mkdir Weights
D:\> mkdir Pickle
```

### Step 5: Copy Pickle Files

Copy the following pickle files to the `Pickle` directory:

- `mMIMO_AS_training_data_20000_80_H_HTH_ORG_1D.pickle`
- `mMIMO_AS_training_data_20000_80_labelVal.pickle`
- `mMIMO_AS_training_data_20000_80_labelVal01.pickle`
- `mMIMO_AS_training_data_20000_seed79_ver2_sorted_ranking.pickle`
- `mMIMO_AS_training_data_20000_seed80_ver2_sorted_ranking.pickle`

### Step 6: Download Python Program

Download the Python script from the following GitHub repository:

- [NNV GitHub Repository](https://github.com/kwanghoon/NNV/)

Make sure to download the file named `mMIMO-80 mf - FC HTH for Validation.py`.

### Step 7: Start Training

Run the training script using the following command:

```shell
D:\> python "mMIMO-80 mf - FC HTH for Validation.py"
```

### Training Output

Below is a sample of the expected output during the training process:

```
short-hard-normal
Additional String : | mMIMO FC H hard short|
select 8 of 16
HTH-1D
2024-05-24 10:43:46.931105: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-05-24 10:43:47.783876: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
*** GPU DISABLED" ***
2.16.1
2024-05-24 10:43:49.781573: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 8316244252516011550
xla_global_id: -1
]
sigmoid
Pickle/mMIMO_AS_training_data_20000_80_H_HTH_ORG_1D.pickle
Pickle/mMIMO_AS_training_data_20000_80_labelVal.pickle
Pickle/mMIMO_AS_training_data_20000_80_labelVal01.pickle
1560000
40000
(1600000, 256)
(1600000, 16)
(1600000, 16)

...

Total params: 378,625 (1.44 MB)
Trainable params: 376,629 (1.44 MB)
Non-trainable params: 1,996 (7.80 KB)
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 913us/step - accuracy: 0.1182 - loss: 0.4491 - precision: 0.7794
==========================================
Pruned Accuracy:
- Accuracy :  0.11940000206232071
- Loss     :  0.44822415709495544
- Precision:  0.779856264591217
==========================================
```