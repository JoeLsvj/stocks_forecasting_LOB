import pandas as pd
import numpy as np
import torch
import tabulate
import tqdm.notebook as tqdm
import Dataset


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
            # Move the data to the device
            X, y = X.to(device), y.to(device)
            
            # Reset the gradients
            optimizer.zero_grad()
            
            # Compute the prediction
            y_pred = model(X)
            
            # Compute the loss
            loss = criterion(y_pred, y)
            
            # Update the model
            loss.backward()
            optimizer.step()
        
        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            loss_train = 0
            n_elements = 0
            for X, y in train_loader:
                # Move the data to the device
                X, y = X.to(device), y.to(device)
                
                # Compute the prediction
                y_pred = model(X)
                
                # Compute the loss
                loss = criterion(y_pred, y)
                
                # Update the loss
                loss_train += loss.item()
                n_elements += 1
            loss_train /= n_elements
            losses_train.append(loss_train)
            
            loss_val = 0
            n_elements = 0
            for X, y in val_loader:
                # Move the data to the device
                X, y = X.to(device), y.to(device)
                
                # Compute the prediction
                y_pred = model(X)
                
                # Compute the loss
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
    

class Metrics:
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def sMAPE(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))




class ModelTools:
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def save_model(model, path, full=False):
        if full:
            torch.save(model, path)
        else:
            torch.save(model.state_dict(), path)

    @staticmethod
    def build_model(model_class, params):
        model = model_class(**params)
        return model

    @staticmethod
    def ht_train_wrapper(dataset_train, dataset_val, model_class, params, device):
        batch_size = params['batch_size']
        n_epochs = params['n_epochs']
        lr = params['lr']
        # Delete the keys
        del params['batch_size']
        del params['n_epochs']
        del params['lr']
        # Build the data loaders
        train_loader = Dataset.TimeSeriesLoader(
            dataset = dataset_train,
            batch_size = batch_size,
            shuffle=True
        )
        val_loader = Dataset.TimeSeriesLoader(
            dataset = dataset_val,
            batch_size = batch_size,
            shuffle=False
        )
        # Build the model
        model = ModelTools.build_model(model_class, params).to(device)
        # Build the optimizer and the loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        # Train the model
        results = train(
            train_loader = train_loader,
            val_loader = val_loader,
            model = model,
            optimizer = optimizer,
            criterion = criterion,
            device = device,
            n_epochs = n_epochs
        )
        # Average the last 5 losses on the validation set
        avg_loss = np.mean(results['losses_val'][-5:])
        # Return negative loss, since the bayesian optimization is a maximization problem
        return -avg_loss


class DataTools:
    
    def __init__(self) -> None:
        pass
    
    
    @staticmethod
    def split_x_y(df, target):
        X = df.drop(target, axis=1)
        y = df[target]
        return X.values, y.values
    
    @staticmethod
    def numpy_to_tensor(*args, device, dtype):
        return [torch.tensor(arg, dtype=dtype).to(device) for arg in args]
    
    @staticmethod
    def train_val_test_split(df, train_size=0.7, val_size=0.15, test_size=0.15):
        assert train_size + val_size + test_size == 1, 'The sum of the sizes must be equal to 1'
        assert train_size > 0 and val_size > 0 and test_size > 0, 'The sizes must be greater than 0'
        
        split_idx = int(train_size * len(df))
        train = df.iloc[:split_idx].copy()
        split_idx += int(val_size * len(df))
        val = df.iloc[split_idx-int(val_size * len(df)):split_idx].copy()
        test = df.iloc[split_idx:].copy()
        
        return train, val, test
    

    # Function which returns the cross fold validation sets for the fi-2010 dataset
    # 0 -> 1, 2
    # 1 -> (1+2), 3
    # 2 -> (1+2+3), 4
    # ...
    @staticmethod
    def cross_fold_timeseries(df):
        # create generator
        days = df['Day'].unique().tolist()
        for i,day in enumerate(days[1:]):
            print('Day:', day)
            train_set, val_set, test_set = None, None, None
            
            if i == 0:
                # Case in which there are only two days
                test = df[df['Day'] == day]
                tmp = df[df['Day'] < day]
                idx_split = int(0.7 * len(tmp))
                train = tmp.iloc[:idx_split]
                val = tmp.iloc[idx_split:]
                
                train_set = train
                val_set = val
                test_set = test            
            else:
                # Case in which there are more than two days
                test = df[df['Day'] == day]
                tmp = df[df['Day'] < day-1]
                middle_day = df[df['Day'] == day-1]
                idx_split = int(0.7 * len(middle_day))
                train = pd.concat([tmp, middle_day.iloc[:idx_split]])
                val = middle_day.iloc[idx_split:]
                
                train_set = train
                val_set = val
                test_set = test
            
            yield train_set, val_set, test_set
        

        
if __name__ == '__main__':
    # Test the cross_fold_timeseries function
    df = pd.read_parquet('../Data/FI-2010-Cleaned/NoAuction/Zscore/KESBV.parquet')
    for train_set, val_set, test_set in DataTools.cross_fold_timeseries(df):
        print('Train:', train_set['Day'].unique())
        print('shape:', train_set.shape)
        print('Val:', val_set['Day'].unique())
        print('shape:', val_set.shape)
        print('Test:', test_set['Day'].unique())
        print('shape:', test_set.shape)
        print()