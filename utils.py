import os
import sys
import time
import pickle
import torch
import numpy as np
#from tqdm import tqdm
from sklearn.metrics import r2_score, roc_auc_score


def make_progress_bar(counter, iterations, suffix=''):
    bar_length = 50
    filled_up_Length = int(round(bar_length* counter / float(iterations)))
    percentage = round(100.0 * counter/float(iterations),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s %s\n' %(bar, percentage, '%', suffix))
    sys.stdout.flush()


def get_performance(dataloader, model, accname, loss, regression:bool):
    # the function returns the accuracy of the network on 
    # a certain dataset/dataloader
    
    total_error = 0.
    total_accuracy = 0.
    total_steps = 0
    # torch.no_grad() is used only for the assessing/validation phase only. 
    # When you only perform validation, gradients are not needed for backward phase.
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            error = loss(outputs, labels)
            if regression:
                # R^2 regression metric
                accuracy = r2_score(labels, outputs)
            else:
                # classification metrics
                accuracy = torch.count_nonzero(torch.argmax(outputs, dim = 1) == labels).item()/labels.size(0)
                # or roc_auc metric
            total_steps += 1
            total_error += error
            total_accuracy += accuracy
    print(accname + f'error: {total_error/total_steps}')
    #print(accname + f'accuracy: {total_accuracy/total_steps}')
    # mean of losses for each batch
    return total_error/total_steps, total_accuracy/total_steps
    

def batch_gd(model_id, model, criterion, optimizer, trainloader, testloader, epochs, regression:bool, save:bool):

    training_info = {
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': []
    }
    best_test_loss = np.inf
    best_test_epoch = 0
    print("Start training...\n")

    for epoch in range(epochs):
        make_progress_bar(epoch, epochs, suffix=f' starting epoch {epoch + 1}')
        # train the model
        model.train()
        train_loss = []
        train_accuracy = []
        start = time.time()
        for inputs, labels in trainloader:
            # move data to GPU, if available
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            print(labels.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            if not regression:
                labels = torch.squeeze(labels)
            loss = criterion(outputs, labels)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            # store the performances on the train set directly during the training epochs
            train_loss.append(loss.item())
            if regression:
                # R^2 regression metric
                snap_accuracy = r2_score(labels.detach().numpy(), outputs.detach().numpy())
            else:
                # classification metrics
                snap_accuracy = torch.count_nonzero(torch.argmax(outputs, dim = 1) == labels).item()/labels.size(0)
            train_accuracy.append(snap_accuracy)
        end = time.time()
        print(f'Duration of training epoch {epoch + 1}: {end - start}, seconds')
        # Get train loss and test loss
        train_loss = np.mean(train_loss, dtype=np.float64)
        train_accuracy = np.mean(train_accuracy, dtype=np.float64)

        # evaluate the model after an epoch of training
        model.eval()
        test_loss = []
        test_accuracy = []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)
                if not regression:
                    labels = torch.squeeze(labels)
                loss = criterion(outputs, labels)
                test_loss.append(loss.item())
                if regression:
                    # R^2 regression metric
                    snap_accuracy = r2_score(labels, outputs)
                else:
                    # classification metrics
                    snap_accuracy = torch.count_nonzero(torch.argmax(outputs, dim = 1) == labels).item()/labels.size(0)
                test_accuracy.append(snap_accuracy)
        test_loss = np.mean(test_loss, dtype=np.float64)
        test_accuracy = np.mean(test_accuracy, dtype=np.float64)
        # save the results after each training epoch
        training_info['train_loss'].append(train_loss)
        training_info['test_loss'].append(test_loss)
        training_info['train_accuracy'].append(train_accuracy)
        training_info['test_accuracy'].append(test_accuracy)

        # evaluate the performances both on train and test sets after each training epoch
        #train_loss, train_accuracy = get_performance(dataloader=trainloader, model=model, accname = "train_", loss=criterion, regression=True)
        #training_info['train_loss'].append(train_loss)
        #training_info['train_accuracy'].append(train_accuracy)
        #test_loss, test_accuracy = get_performance(dataloader=testloader, model=model, accname = "test_", loss=criterion, regression=True)
        #training_info['test_loss'].append(test_loss)
        #training_info['test_accuracy'].append(test_accuracy)

        saving_path = './saved_models/' + model_id
        if (test_loss < best_test_loss and save):
            torch.save(model, os.path.join(saving_path, 'best_model.pt'))
            best_test_loss = test_loss
            best_test_epoch = epoch
            print('model saved')

        print(f'Epoch {epoch + 1}/{epochs}, '
              f'Train Loss: {train_loss:.8f}, Train Accuracy: {train_accuracy: .8f}, '
              f'Validation Loss: {test_loss:.8f}, Validation Accuracy: {test_accuracy: .8f}, ')
    
    # save the final results after the training
    if save:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, os.path.join(saving_path, 'checkpoint.pt'))

        with open(os.path.join(saving_path, 'training_process.pkl'), 'wb') as f:
            pickle.dump(training_info, f)

    return training_info
