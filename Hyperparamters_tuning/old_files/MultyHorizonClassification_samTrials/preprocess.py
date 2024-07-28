import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import logging

# This function is used to prepare the input data for the model
def prepare_x(data): # the input is the data of size (40, 5) whwre 40 is the number of features and 5 is the number of time steps
    df1 = data[:40, :].T 
    return np.array(df1) # return the input data of size (40, 5) where 40 is the number of features and 5 is the number of time steps


# This function is used to prepare the output data for the model
def get_label(data): # the data has size (40, 5) where 40 is the number of features and 5 is the number of time steps
    lob = data[-5:, :].T 
    all_label = []

    for i in range(lob.shape[1]): # this for loop is used to prepare the output data for the model
        one_label = lob[:, i] - 1
        one_label = keras.utils.to_categorical(one_label, 3)
        one_label = one_label.reshape(len(one_label), 1, 3)
        all_label.append(one_label)

    return np.hstack(all_label) # return the output data of size (5, 1, 3), where 5 is the number of time steps, 1 is the number of features, and 3 is the number of classes

# This function is used to prepare the input data for the model
# X is the input data of size (40, 5) where 40 is the number of features and 5 is the number of time steps
# Y is the output data of size (5, 1, 3), where 5 is the number of time steps, 1 is the number of features, and 3 is the number of classes
# T is the number of time steps
def data_classification(X, Y, T): 
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    # The output is 
    # the input data of size (36, 5, 40) where 36 is the number of samples, 5 is the number of time steps, and 40 is the number of features
    # the output data of size (36, 5, 1, 3), where 36 is the number of samples, 5 is the number of time steps, 1 is the number of features, and 3 is the number of classes
    return dataX.reshape(dataX.shape + (1,)), dataY 


# This function creates the dataset for the model
# x_train is the input data of size (36, 5, 40) where 36 is the number of samples, 5 is the number of time steps, and 40 is the number of features
# y_train is the output data of size (36, 5, 1, 3), where 36 is the number of samples, 5 is the number of time steps, 1 is the number of features, and 3 is the number of classes
# batch_size is the batch size
# method is the method to create the dataset
# shuffle is a boolean variable to shuffle the dataset
def create_dataset(x_train, y_train, batch_size, method, shuffle=False):
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if shuffle:
        train_ds = train_ds.shuffle(len(x_train))
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))

    if method == 'train':
        return train_ds.repeat()

    if method == 'val':
        return train_ds

    if method == 'test':
        return train_ds

    if method == 'prediction':
        train_ds = tf.data.Dataset.from_tensor_slices((x_train))
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
        train_ds = train_ds.map(lambda d: (tf.cast(d, tf.float32)))
        return train_ds

# This function is used to evaluate the model
# real_y is the real output data of size (36, 5, 1, 3), where 36 is the number of samples, 5 is the number of time steps, 1 is the number of features, and 3 is the number of classes
# pred_y is the predicted output data of size (36, 5, 1, 3), where 36 is the number of samples, 5 is the number of time steps, 1 is the number of features, and 3 is the number of classes
def evaluation_metrics(real_y, pred_y):
    real_y = real_y[:len(pred_y)]
    logging.info('-------------------------------')

    for i in range(real_y.shape[1]):
        print(f'Prediction horizon = {i}')
        print(f'accuracy_score = {accuracy_score(np.argmax(real_y[:, i], axis=1), np.argmax(pred_y[:, i], axis=1))}')
        print(f'classification_report = {classification_report(np.argmax(real_y[:, i], axis=1), np.argmax(pred_y[:, i], axis=1), digits=4)}')
        print('-------------------------------')

# This function is used to prepare the input data for the decoder
# data is the output data of size (36, 5, 1, 3), where 36 is the number of samples, 5 is the number of time steps, 1 is the number of features, and 3 is the number of classes
# teacher_forcing is a boolean variable to use teacher forcing
def prepare_decoder_input(data, teacher_forcing):
    if teacher_forcing:
        first_decoder_input = keras.utils.to_categorical(np.zeros(len(data)), 3)
        first_decoder_input = first_decoder_input.reshape(len(first_decoder_input), 1, 3)
        decoder_input_data = np.hstack((data[:, :-1, :], first_decoder_input))

    if not teacher_forcing:
        decoder_input_data = np.zeros((len(data), 1, 3))
        decoder_input_data[:, 0, 0] = 1.

    # The output is the input data for the decoder of size (36, 5, 3), where 36 is the number of samples, 5 is the number of time steps, and 3 is the number of classes
    return decoder_input_data