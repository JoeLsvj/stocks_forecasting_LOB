from audioop import mul
from this import d
import time 
import numpy as np
import pandas
import ccxt     #order maker
import os
import sys
import config
import ta       #indicators library. Works with numpy and pandas
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator#, StochRSIIndicator     #the close is moving in timestamp. the overture not.
from ta.trend import EMAIndicator, AroonIndicator#, KSTIndicator, MACD
from ta.volume import MFIIndicator, ChaikinMoneyFlowIndicator#, VolumePriceTrendIndicator
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
import torch.nn as nn
import torch
import torch.nn.functional as F
import json, argparse
import random as rn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
#import tensorboard as tb
import torch.utils
import os
import time


