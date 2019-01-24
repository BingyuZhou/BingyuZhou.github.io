---
layout: post
title:  "Install GPU-enabled Tensorflow with Ubuntu 18.04"
date:   2018-04-28 16:00:00
categories: Software
---

Ubuntu 18.04 is released recently. Many developers will adapt the Tensorflow development environment with the new GNOME style's Ubuntu OS. Here I show my successful installation of Tensorflow with GPU in Ubuntu 18.04. Hope it can help you out of the mess for multiple platforms' compatability.

## Platforms

- Ubuntu 18.04

- Tensorflow 1.7

- CUDA 9.1

- cuDNN v7.1.3 for CUDA 9.1

- Anaconda (Python 3.6)

## Ubuntu 18.04

For the installation of Ubuntu 18.04, please refer [here](https://tutorials.ubuntu.com/tutorial/tutorial-install-ubuntu-desktop?_ga=2.151141374.653516307.1524945589-1651284035.1524945589#0).

## Anaconda

Anaconda is a platform designed for data scientists. It includes all essential tools you need for machine learning and data analysis, such as Jupyter, Spyder, VScode and so on. Meanwhile, it also has the feature to create environments similar with the *virtualenv*.

Follow the official installation [here](https://www.anaconda.com/download/#linux). You will have it immediately. I choose Python 3.6 version.

## Verify CUDA-Capable GPU

First, we need to make sure you have a CUDA-capable GPU. Run the command

```
lspci | grep -i nvidia
```

it will print the available Nvidia GPU type if you have one.

## Install CUDA

Once you have a capable GPU, it is ready to proceed the preparation of GPU for tensorflow.

CUDA is the fundamental parallel computing platform to enable the full power of GPU.

For Ubuntu 18.04, there are two ways to install CUDA. The standard way is to download the CUDA files from [Nvidia website](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=deblocal). I choose the latest CUDA version 9.1 in deb installation type. And inistall it

```
sudo dpkg -i cuda-repo-ubuntu1704-9-1-local_9.1.85-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

After this, you can see there will be CUDA files under `/usr/local/`.

Here is the most important thing after installing CUDA. We have to add the path to environment variables. I add the paths at the end of `~/.bashrc` file.

```
# path for  CUDA and Tensorflow
export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Another faster installation approach is to use the third prty service in ubuntu:

```
sudo apt install nvidia-cuda-toolkit
```

This will install the latest cuda 9.1 version. But the installation path will be different. The CUDA files will be under `/usr/lib`. Therefore, when you add paths, you have to adapt the path to cuda.

## Install cuDNN 

cuDNN should be compatible with the version of CUDA. You can download cuDNN [here](https://developer.nvidia.com/rdp/cudnn-download). I choose cuDNN v7.1.3 for CUDA 9.1. We just need to copy cuDNN files to certain locations under CUDA.

```
tar -zxvf cudnn-9.1-linux-x64-v7.1.3.tgz 
sudo cp cuda/lib64/* /usr/local/cuda/lib64/
sudo cp cuda/include/cudnn.h /usr/lib/cuda/include
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h
sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*
```

## Install libcupti-dev

This is easy.

```
sudo apt install libcupti-dev
```

And add the path to environment variables in `~/.bashrc`

```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64
```

## Tensorflow

Now GPU is ready! Time to install Tensorflow. I use Anaconda to create environment for tensorflow. And then install Tensorflow under that environment.

```
source ~/.bashrc
conda create -n tf pip python=3.6
source activate tf
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.7.0-cp36-cp36m-linux_x86_64.whl
```

Now Tensorflow 1.7 is ready. Enjoy it!







