import pandas as pd
import numpy as np

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
    