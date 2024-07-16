import torch
import numpy as np
import pandas as pd


class TimeSeriesDataset(torch.utils.data.Dataset):
    
    @staticmethod
    #def build_windows(X:np.array, y:np.array, seq_len_x:int=10, seq_len_y:int=1) -> tuple:
    def build_windows(X:torch.tensor, y:torch.tensor, seq_len_x:int=10, seq_len_y:int=1) -> tuple:
        seq_len = max(seq_len_x, seq_len_y)
        tot_len = X.shape[0] - seq_len
        
        X_windows = X.unfold(0, seq_len_x, 1).permute(0,2,1)[0:tot_len]
        y_windows = y.unfold(0, seq_len_y, 1).permute(0,2,1)[0:tot_len]
        
        return X_windows, y_windows
        
    
    def __init__(
        self,
        X : pd.DataFrame,
        y : pd.DataFrame,
        seq_len_x : int = 10,
        seq_len_y : int = 1,
        offset : int = 0,
        channels: bool = False
    ) -> None:
        self.offset  = offset
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y
        # Convert original time series into a windowed time series
        self.X, self.y = self.build_windows(X, y, seq_len_x, seq_len_y)
        # Compute the length of the dataset
        self.length  = self.X.shape[0] - self.offset
        self.channels = channels
        # Save the shape of the dataset
        self.shape = {
            'X': self.X.shape,
            'y': self.y.shape
        }
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx) -> tuple:
        if not self.channels:
            return self.X[idx], self.y[idx+self.offset]
        else:
            return np.expand_dims(self.X[idx], axis=0), np.expand_dims(self.y[idx+self.offset], axis=0)


if __name__ == '__main__':
    df = pd.read_parquet('../Data/FI-2010-Cleaned/NoAuctionZscore/WRT1V_NoAuction_Zscore.parquet')
    X,y = df.drop(columns=['Label_10']).values, df['Label_10'].values
    y = y.reshape(-1,1)
    dataset = TimeSeriesDataset(X, y, seq_len=10, offset=0, channels=True)
    
    print(dataset.X.shape, dataset.y.shape)
    
    print('One iteration')
    for x,y in dataset:
        print(x.shape, y.shape)
        break
    


class TimeSeriesLoader(torch.utils.data.DataLoader):
    
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False
    ) -> None:
        super(TimeSeriesLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def __iter__(self):
        for batch in super(TimeSeriesLoader, self).__iter__():
            yield batch
            