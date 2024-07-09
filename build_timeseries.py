import torch
import numpy as np
import pandas as pd


class SlidingWindowDataset(object):
	# The function implements the 'sliding window' method, in order to to create 
	# the actual dataset to use with the neural network models. This method creates
	# the features cutting the dataset with a window of length 'lookback_period', 
	# which slides down by one unit each time. The corresponding target for that 
	# window/feature is the next value (or values for multi-horizon prediction)
	# of the targets' columns (there can be 1 up to m target columns)
	# The dataset can be then processed to obtain train and test datasets, 
	# and to build an iterator on it, eventually converting the elements 
	# in torch.tensor, grouped in batches of batch_size.
	# The function works both passing the raw_dataset to the __init__ function, 
	# and passing the root directory where to find a saved raw_dataset to load.
	
	def __init__(
		self, 
		split_rate:float = 0.7, 
		lookback_period:int = 100,
		#prediction_horizon = 1,
		target_columns = None,
		channels:bool = False,
		raw_dataset = None,
		root_dir:str = None
		):

		self.table_dataset = []

		# create the targets:
		if (type(target_columns)!=list):
			targets = target_columns[lookback_period:]
		else:
			targets = raw_dataset[target_columns][lookback_period:]
		#targets = np.array(targets, dtype=np.int64)
		targets = np.array(targets, dtype=np.float32)

		# create the features with the sliding window method		
		features = []
		# older rows on top (lower index)
		for i in range(0, len(raw_dataset)-lookback_period, 1):
			dump = np.array(raw_dataset[i:i+lookback_period:1], dtype=np.float32)
			# add channels dimension in the tensor when needed (for example in CNN models)
			# the general pytorch tensor structure is [batch size, channels, heigh, width]
			# for LSTM-only models the channels dimension is not needed and the argument has to be false
			if (channels):	dump = np.expand_dims(dump, axis=0)
			features.append(dump)
		for i in range(len(features)):
			self.table_dataset.append((features[i], targets[i]))
			# append the targets according to the prediction horizon selected			
			#self.table_dataset.append((features[i], targets[i:i+prediction_horizon]))
		
		self.complete = self.table_dataset
		self.valid    = self.table_dataset[int(split_rate*len(self.table_dataset)):]
		self.train    = self.table_dataset[:int(split_rate*len(self.table_dataset))]


class CreateLoader(torch.utils.data.Dataset):
	def __init__(self, dataset = None, transform = None):
		super().__init__()
		# 'transform' (callable, optional): Optional transform to be applied on a sample.
		self.table_dataset = dataset
		self.transform = transform

	def __len__(self):
		return len(self.table_dataset)

	# build the iterator function in order to mount a dataloader -> useful for batching data with pytorch
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		table, label = self.table_dataset[idx]
		# convert to tensor in order to be processed by pytorch nn modules
		table = torch.tensor(table)
		label = torch.tensor(label)
		#label = torch.squeeze(label)
		sample = [table, label]
		if self.transform:
			sample[0] = self.transform(sample[0])

		return sample


