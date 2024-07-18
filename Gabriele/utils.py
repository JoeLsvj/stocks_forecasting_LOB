import pandas as pd
import numpy as np
import torch
import tabulate
import tqdm.notebook as tqdm


class FeaturesExtractionLOB:
    
    def __init__(self, data : pd.DataFrame):
        self.data = data
    
    def extract(self):
        # Mid price
        self.data['MP'] = (self.data['P_Ask_1'] + self.data['P_Bid_1']) / 2
        # Wheigted mid price
        self.data['WMP'] = (self.data['P_Ask_1'] * self.data['V_Ask_1'] + self.data['P_Bid_1'] * self.data['V_Bid_1']) / (self.data['V_Ask_1'] + self.data['V_Bid_1'])
        # Bid-Aks spread
        self.data['Spread'] = self.data['P_Ask_1'] - self.data['P_Bid_1']
        # Order Book Imbalance
        self.data['Imbalance'] = (self.data['V_Ask_1'] - self.data['V_Bid_1']) / (self.data['V_Ask_1'] + self.data['V_Bid_1'])
        # Bid, Ask and Total Volume
        self.data['Depth_Bid'] = self.data.filter(regex='V_Bid').sum(axis=1)
        self.data['Depth_Ask'] = self.data.filter(regex='V_Ask').sum(axis=1)
        self.data['Depth_Total'] = self.data['Depth_Bid'] + self.data['Depth_Ask']
        
        return self.data


def train(
    train_loader : torch.utils.data.DataLoader,
    val_loader   : torch.utils.data.DataLoader,
    model        : torch.nn.Module,
    optimizer    : torch.optim.Optimizer,
    criterion,
    device       : torch.device,
    n_epochs     : int,
):
    progress_bar = tqdm.tqdm(range(n_epochs), desc='Training')
    losses_train = []
    losses_val = []
    
    for epoch in progress_bar:
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        
        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            loss_train = 0
            n_elements = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                loss_train += loss.item()
                n_elements += 1
            loss_train /= n_elements
            losses_train.append(loss_train)
            
            loss_val = 0
            n_elements = 0
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                loss_val += loss.item()
                n_elements += 1
            loss_val /= n_elements
            losses_val.append(loss_val)
        
        # Update the progress bar    
        progress_bar.set_postfix({'loss_train': loss_train, 'loss_val': loss_val})
        
    return {
        'model': model,
        'losses_train': losses_train,
        'losses_val': losses_val
    }


def evaluate(
    loader   : torch.utils.data.DataLoader,
    model    : torch.nn.Module,
    criterion,
    device   : torch.device,
):
    true = []
    pred = []
    losses = []
    
    model.eval()
    with torch.no_grad():
        loss = 0
        n_elements = 0
        for X, y in loader:
            # Move the data to the device
            X, y = X.to(device), y.to(device)
            # Compute the prediction
            y_pred = model(X)
            
            # Reshape 128,1,1 to 128,1
            y_pred = y_pred.squeeze()
            y = y.squeeze()
            
            
            # Save the true and predicted values
            true.append(y.cpu().numpy())
            pred.append(y_pred.cpu().numpy())
                        
            # Compute the loss
            loss = criterion(y_pred, y)
            losses.append(loss.item())
    
    return {
        'true': true,
        'pred': pred,
        'losses': losses
    }
    
def sMAPE(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))