# Simplifying Temporal Heterogeneous Network for Continuous-Time Link Prediction

This anonymous repo provides an implementation of **STHN** for CIKM 2023 submission.

## Requirements

```
# create conda virtual enviroment
conda create --name sthn python=3.9

# activate environment
conda activate sthn

# pytorch 
pip install torch==1.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# pytorch-geometric
pip install torch-geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

# pybind11
pip install pybind11

# torchmetrics
pip install torchmetrics==0.11.0
```

## Datasets
MathOverflow can be found in [Stanford Snap Datasets](https://snap.stanford.edu/data/sx-mathoverflow.html).

You can access the original Netflix dataset [here](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).

We provide the Movielens data for testing in /DATA/.

## Run the code

### 1. Compile C++ smapler

```shell
python setup.py build_ext --inplace
```
### 1. Data pre-process

```shell
python gen_grapgh.py --data movie
```
### 2. Link prediction

```shell
python train.py --movie --max_edges 50
```
### 3. Link type prediction

```shell
python train.py --movie --max_edges 50 --predict_class
```
