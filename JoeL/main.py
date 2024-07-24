import time 
import pandas
import random
import json, argparse

# ML
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, roc_auc_score

# pytorch
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils

# software modules
import build_timeseries
import LSTM
import CNN_LSTM
import utils



dataset = pd.read_csv('./LOBSTER/MSFT_2012-06-21_34200000_57600000_orderbook_10.csv', header=None)
dataset.columns = [ 'P_Ask_1', 'V_Ask_1', 'P_Bid_1', 'V_Bid_1', 'P_Ask_2', 'V_Ask_2',
                    'P_Bid_2', 'V_Bid_2', 'P_Ask_3', 'V_Ask_3', 'P_Bid_3', 'V_Bid_3',
                    'P_Ask_4', 'V_Ask_4', 'P_Bid_4', 'V_Bid_4', 'P_Ask_5', 'V_Ask_5',
                    'P_Bid_5', 'V_Bid_5', 'P_Ask_6', 'V_Ask_6', 'P_Bid_6', 'V_Bid_6',
                    'P_Ask_7', 'V_Ask_7', 'P_Bid_7', 'V_Bid_7', 'P_Ask_8', 'V_Ask_8',
                    'P_Bid_8', 'V_Bid_8', 'P_Ask_9', 'V_Ask_9', 'P_Bid_9', 'V_Bid_9',
                    'P_Ask_10', 'V_Ask_10', 'P_Bid_10', 'V_Bid_10']

dataset = dataset[:]
# compute the price with the 1st level of the LOB and add the column to the dataset
#dataset['mid_price'] = (dataset['P_Ask_1'] + dataset['P_Bid_1']) / 2
# The StandardScaler scales (to unit variance and removing the mean) 
# each feature column -> features processing
scaler = preprocessing.StandardScaler()
scaler.set_output(transform = 'pandas')
dataset = scaler.fit_transform(dataset)


for offset in [10, 20, 30, 50, 100]:
    ### dataset creation
    # the final time-series dataset
    lookback_period = 100
    # Remember to reshape the arrays in order to get the correct tensor structure for convolution: [batch, channels, height, width]
    # The standard tensor structure for LSTM layers is [batch, lookback_period, number of features] = [batch, height, width]
    Dataset  =  build_timeseries.SlidingWindowDataset(	raw_dataset = dataset, 
    													split_rate = 0.7,
    													#target_columns = ['P_Ask_1', 'V_Ask_1', 'P_Bid_1', 'V_Bid_1'],
    													#target_columns = ['mid_price'],
    													target_columns = ['P_Ask_1'],
    													#target_columns = targets,
    													lookback_period = lookback_period,
    													offset = offset,
    													channels = False
    												)
    # in the following lines the features are transformed into torch tensors, within the train and test loaders
    trainset =  build_timeseries.CreateLoader(dataset = Dataset.train, transform = None)
    testset  =  build_timeseries.CreateLoader(dataset = Dataset.valid, transform = None)
    # dataloaders -> the batch dimension is added as first
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True, drop_last=False)
    testloader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=32, shuffle=True, drop_last=False)


    #### select the model
    W, H = len(dataset.columns), len(dataset)
    model = LSTM.LSTM1(input_size=W, hidden_layer_size=64, num_layers=2, output_size=1, dropout=0.2)
    #model = LSTM.LSTM2(input_size=W, hidden_layer_size=64, num_layers=2, output_size=1, dropout=0.2)
    #model = LSTM.LSTMWithAttention(  input_size=W, 
    #                                 encoder_hidden_size=64, 
    #                                 encoder_layers=2, 
    #                                 decoder_hidden_size=64, 
    #                                 decoder_layers=2,
    #                                 output_size=1,
    #                                 dropout=0.2)

    #model = CNN_LSTM.LobCNN(lookback_period = lookback_period,
    #                        inner_convolutional_layers = 2,
    #                        smoothing_percentage = 0.06,
    #                        convolution_channels = 32,
    #                        inception_channels = 64,
    #                        lstm_layers = 1,
    #                        output_size = 1
    #                    )


    #### train the model
    model.to(model.device)
    learning_rate = 0.001
    epochs = 3
    LOSS = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_info = utils.batch_gd( model_id=f'{model.name}-{offset}-reg', 
                                    model=model, 
                                    criterion=LOSS, 
                                    optimizer=optimizer, 
                                    trainloader=trainloader, 
                                    testloader=testloader, 
                                    epochs=epochs, 
                                    regression=True, 
                                    save=False)


    ### plot loss/accuracy                     
    epochs_list = list(range(0,epochs))
    _, axs = plt.subplots(2, 1, figsize=(16,12))
    axs[0].plot(epochs_list, training_info['test_loss'],  label = "test error")
    axs[0].plot(epochs_list, training_info['train_loss'], label = "train error")
    axs[0].legend(loc = 'best')
    axs[1].plot(epochs_list, training_info['test_accuracy'],  label = "test accuracy")
    axs[1].plot(epochs_list, training_info['train_accuracy'], label = "train accuracy")
    axs[1].legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(f'./figures/{model.name}-{offset}-LOB-reg-LOSS.png')


    ### plot figures
    with torch.no_grad():
        out = list(range(len(Dataset.complete)))
        features = list(range(len(Dataset.complete)))
        for i in range(len(Dataset.complete)):
            features[i] = torch.Tensor(Dataset.complete[i][0])
            features[i] = torch.unsqueeze(features[i], axis = 0)
            features[i] = features[i].to(model.device)
            out[i] = model(features[i]).cpu().squeeze().numpy()

    #targets = dataset[lookback_period:]['mid_price']
    targets = dataset[lookback_period:]['P_Ask_1']
    #targets = targets[lookback_period:]

    plt.figure(figsize=(30, 15))
    plt.plot(targets[:int(len(targets)*0.7)], label = "Actual asset price: train", color = "navy")
    plt.plot(targets[int(len(targets)*0.7):], label = "Actual asset price: test", color = "purple")
    plt.plot(list(range(lookback_period-1+offset,H)), out, label = "Predicted asset price", color = "green")
    plt.title("Asset price: actual vs predicted")
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.legend(loc = 'best', prop={'size': 10})
    plt.tight_layout()
    plt.savefig(f'./figures/{model.name}-{offset}-LOB-reg-PRYCE.png')

    fig = go.Figure()
    fig.add_scatter(y=targets[:int(len(targets)*0.7)], mode='lines', name= "Actual asset price: train", line_color = "navy")
    fig.add_scatter(x=list(range(int(len(targets)*0.7), int(len(targets)))), y=targets[int(len(targets)*0.7):], mode='lines', name= "Actual asset price: test",  line_color = "purple")
    fig.add_scatter(x=list(range(lookback_period-1+offset,H)), y=out, mode='lines', name= "Predicted asset price", line_color = "green")
    fig.write_html(f'./figures/{model.name}-{offset}-LOB-reg-pryce.html')




