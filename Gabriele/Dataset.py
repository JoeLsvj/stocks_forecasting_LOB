import torch
import numpy as np
import pandas as pd


class TimeSeriesDataset(torch.utils.data.Dataset):
    
    @staticmethod
    def build_windows(X:np.array, y:np.array, seq_len:int=10, offset:int=0) -> tuple:
        shape_X = (X.shape[0] - seq_len + 1, seq_len, X.shape[1])
        shape_y = (y.shape[0] - seq_len + 1, y.shape[1])
        stride_X = (X.strides[0], X.strides[0], X.strides[1])
        stride_y = (y.strides[0], y.strides[1])
        
        X = np.lib.stride_tricks.as_strided(X, shape_X, stride_X)
        y = np.lib.stride_tricks.as_strided(y, shape_y, stride_y)
        
        return X, y
        
    
    def __init__(
        self,
        X : pd.DataFrame,
        y : pd.DataFrame,
        seq_len : int = 10,
        offset : int = 0,
        channels: bool = False
    ) -> None:
        self.offset  = offset
        self.seq_len = seq_len
        self.length  = len(X) - self.offset - self.seq_len
        self.channels = channels

        self.X, self.y = self.build_windows(X, y, seq_len, offset)

    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx) -> tuple:
        if not self.channels:
            return self.X[idx], self.y[idx+self.offset]
        else:
            return np.expand_dims(self.X[idx], axis=0), self.y[idx+self.offset]


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
            