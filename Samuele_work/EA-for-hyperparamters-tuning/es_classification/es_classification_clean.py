#!/usr/bin/env python
# coding: utf-8

# # ES for ht for classification 
# 
# HT to tune 
# - lr 
# - dropout 
# - number of units 
# - activation function 
# 
# 
# 

# ## The NN

# In[10]:


# Libs

import torch # the main pytorch library
from torch import nn # the sub-library containing neural networks
from torch.utils.data import DataLoader, Subset # an object that generates batches of data for training/testing
from torchvision import datasets # popular datasets, architectures and common image transformations
from torchvision.transforms import ToTensor # an easy way to convert PIL images to Tensors

import pandas as pd
import numpy as np # NumPy, the python array library
import random # for generating random numbers
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
import time
import csv

# Custom modules
import Models_class as Models
import Dataset
import utils

# Set seeds

SEED = 0 # seed for reproducibility

random.seed(SEED) # set the seed for random numbers
np.random.seed(SEED) # set the seed for numpy arrays


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use the GPU if available
#print(f"default device: {device}")


# In[11]:


df = pd.read_parquet('/home/samuele/Desktop/22_dl/DL_Project/stocks_forecasting_LOB/Data/FI-2010-Cleaned/NoAuction/Zscore/WRT1V.parquet')
df = df.head(500) # For testing purposes (put 20000)
df.drop(columns=['Day'], inplace=True)

target_cols = ['Label_10', 'Label_20', 'Label_30', 'Label_50', 'Label_100']
# Set target cols to dtype int
df[target_cols] = df[target_cols].astype(int)

#df.head()


# In[12]:


logreturn = lambda x: np.log(x) - np.log(x.shift(1))

price_cols = [col for col in df.columns if 'P_' in col]
df[price_cols] = df[price_cols].apply(logreturn, axis=0)
df.dropna(inplace=True)


# In[13]:


# Split the dataframe into train and test
df_train, df_val, df_test = utils.DataTools.train_val_test_split(df)

# Split covariates and response variable(s)
X_train, y_train = utils.DataTools.split_x_y(df_train, target_cols)
X_val, y_val     = utils.DataTools.split_x_y(df_val, target_cols)
X_test, y_test   = utils.DataTools.split_x_y(df_test, target_cols)

# Standardize the covariates
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

# Build tensors 
X_train, X_val, X_test = utils.DataTools.numpy_to_tensor( X_train, X_val, X_test, device=device, dtype=torch.float32 )
y_train, y_val, y_test = utils.DataTools.numpy_to_tensor( y_train, y_val, y_test, device=device, dtype=torch.long )

# Build TimeSeriesDataset
SEQ_LEN_X = 50
SEQ_LEN_Y = 1
OFFSET = 0
dataset_train = Dataset.TimeSeriesDataset(
    X         = X_train,
    y         = y_train,
    seq_len_x = SEQ_LEN_X,
    seq_len_y = SEQ_LEN_Y,
    offset    = OFFSET,
    channels  = False,
    task      = Dataset.TimeSeriesDataset.Task.CLASSIFICATION
)
dataset_val = Dataset.TimeSeriesDataset(
    X         = X_val,
    y         = y_val,
    seq_len_x = SEQ_LEN_X,
    seq_len_y = SEQ_LEN_Y,
    offset    = OFFSET,
    channels  = False,
    task      = Dataset.TimeSeriesDataset.Task.CLASSIFICATION
)
dataset_test = Dataset.TimeSeriesDataset(
    X         = X_test,
    y         = y_test,
    seq_len_x = SEQ_LEN_X,
    seq_len_y = SEQ_LEN_Y,
    offset    = OFFSET,
    channels  = False,
    task      = Dataset.TimeSeriesDataset.Task.CLASSIFICATION
)

# Build TimeSeriesLoader
BATCH_SIZE = 32
dataloader_train = Dataset.TimeSeriesLoader(
    dataset = dataset_train,
    batch_size = BATCH_SIZE,
    shuffle = True
)
dataloader_val = Dataset.TimeSeriesLoader(
    dataset = dataset_val,
    batch_size = BATCH_SIZE,
    shuffle = False
)
dataloader_test = Dataset.TimeSeriesLoader(
    dataset = dataset_test,
    batch_size = 1024,
    shuffle = False
)


# ## Useful commands 
# 

# ## The EA 

# ### Complete algorithm

# In[7]:


import es_classification_functions as esc


# In[14]:


# Example usage
max_fitness_evaluations = 5 # Just for trial
num_epochs_per_nn_evaluation = 1

mu_population_size = 2
init_layersNumber_interval = [1,1]
init_unitsNumber_interval = [10,1000]
init_activation_functions = [nn.ReLU] 
init_lr_interval = (0.0001, 0.1)
init_dropout_interval = (0.0, 0.2)
nn_training_data = dataloader_train
nn_validation_data = dataloader_val
nn_input_size = 40
nn_output_size = 15
device = 'cpu' # 'cuda'
maximize_fitness = False
lambda_offsprings_size = 3
tornamentPercentage_for_offspringGeneration = 0.2
p_crossover_vs_mutation = 0.75
structure_k_value_for_crossover = 0
structure_lb_size = 10
structure_ub_size = np.inf
lr_k = 0
dropout_k = 0
lr_lb = 0.0001
lr_ub = 0.1
dropout_lb = 0.0
dropout_ub = 0.2
csv_file_path = '/home/samuele/Desktop/22_dl/DL_Project/stocks_forecasting_LOB/Samuele_work/EA-for-hyperparamters-tuning/es_classification/csv_results/es_classification2.csv'

best_individual_ever = esc.es_lambdaMu_sld_classification(
    max_fitness_evaluations = max_fitness_evaluations,
    mu_population_size = mu_population_size,
    init_layersNumber_interval = init_layersNumber_interval,
    init_unitsNumber_interval = init_unitsNumber_interval,
    init_activation_functions = init_activation_functions,
    init_lr_interval = init_lr_interval,
    init_dropout_interval = init_dropout_interval,
    nn_training_data = nn_training_data,
    nn_validation_data = nn_validation_data,
    num_epochs_per_nn_evaluation = num_epochs_per_nn_evaluation,
    nn_input_size = nn_input_size,
    nn_output_size = nn_output_size,
    device = device,
    maximize_fitness = maximize_fitness,
    lambda_offsprings_size = lambda_offsprings_size,
    tornamentPercentage_for_offspringGeneration = tornamentPercentage_for_offspringGeneration,
    p_crossover_vs_mutation = p_crossover_vs_mutation,
    structure_k_value_for_crossover = structure_k_value_for_crossover,
    structure_lb_size = structure_lb_size,
    structure_ub_size = structure_ub_size,
    lr_k = lr_k,
    dropout_k = dropout_k,
    lr_lb = lr_lb,
    lr_ub = lr_ub,
    dropout_lb = dropout_lb,
    dropout_ub = dropout_ub,
    csv_file_path=csv_file_path)


# ##
