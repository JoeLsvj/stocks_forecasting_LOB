import numpy as np # for numerical computation

import torch # for deep learning
import torch.nn as nn # for neural network
import torch.nn.functional as F # for neural network functions like activation functions
import torch.optim as optim # for optimization algorithms
from torch.utils.data import DataLoader, Dataset, TensorDataset # for dataloader and dataset

from sklearn.metrics import accuracy_score, classification_report
import logging

def prepare_x(data): # this function is used to prepare the input data
    df1 = data[:40, :].T
    return np.array(df1) # return the input data in the shape of (N, D) where N is the number of samples and D is the number of features


def get_label(data):  # This function is used to prepare the label data
    lob = data[-5:, :].T
    all_label = []

    for i in range(lob.shape[1]):
        one_label = lob[:, i].astype(int) - 1  # Ensure integer type and adjust class indices if necessary
        one_label = F.one_hot(torch.tensor(one_label), num_classes=3).numpy()
        one_label = one_label.reshape(len(one_label), 1, 3)
        all_label.append(one_label)

    return np.hstack(all_label) # return the label data in the shape of (N, T, 3) where N is the number of samples and T is the number of time steps and 3 is the number of classes


def data_classification(X, Y, T): # this function is used to prepare the input and label data
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX.reshape(dataX.shape + (1,)), dataY # return the input and label data in the shape of (N, T, D, 1) and (N, T, 3) respectively where N is the number of samples, T is the number of time steps, D is the number of features and 3 is the number of classes


def create_dataset(x_train, y_train, batch_size, method, shuffle=False): # this function is used to create the dataloader
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = TensorDataset(x_tensor, y_tensor)
    
    if shuffle:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    if method == 'train':
        return dataloader # return the dataloader for training
    elif method == 'val':
        return dataloader # return the dataloader for validation
    elif method == 'test':
        return dataloader # return the dataloader for testing
    elif method == 'prediction':
        x_tensor = torch.tensor(x_train, dtype=torch.float32)
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        return dataloader # return the dataloader for prediction


def evaluation_metrics(real_y, pred_y): # this function is used to evaluate the model, the real_y is the real label data and the pred_y is the predicted label data
    real_y = real_y[:len(pred_y)]
    logging.info('-------------------------------')

    for i in range(real_y.shape[1]):
        print(f'Prediction horizon = {i}')
        print(f'accuracy_score = {accuracy_score(np.argmax(real_y[:, i], axis=1), np.argmax(pred_y[:, i], axis=1))}')
        print(f'classification_report = {classification_report(np.argmax(real_y[:, i], axis=1), np.argmax(pred_y[:, i], axis=1), digits=4)}')
        print('-------------------------------')


def prepare_decoder_input(data, teacher_forcing): # this function is used to prepare the decoder input data
    if teacher_forcing:
        first_decoder_input = F.one_hot(torch.zeros(len(data), dtype=torch.long), num_classes=3).numpy()
        first_decoder_input = first_decoder_input.reshape(len(first_decoder_input), 1, 3)
        decoder_input_data = np.hstack((data[:, :-1, :], first_decoder_input))

    else:
        decoder_input_data = np.zeros((len(data), 1, 3))
        decoder_input_data[:, 0, 0] = 1.

    return decoder_input_data # return the decoder input data in the shape of (N, 1, 3) where N is the number of samples, 1 is the number of time steps and 3 is the number of classes
