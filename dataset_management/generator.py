

import torch
import math
import pandas as pd
import numpy as np


class Sequence2PointGenerator(torch.utils.data.Dataset):
    """
    Class that takes the X values and corresponding Y values. Makes windows of provided sequence length for X and tags along the middle index value of Y
    """
    def __init__(self, data):
        try:
            super().__init__()
        
        except Exception as e:
            print("Error occured in initialization of Sequence2PointGenerator class due to ", e)
            
        finally:
            self.sequence_length = MODEL_CONFIG['SEQUENCE_LENGTH']
            lst = [0] * math.floor(self.sequence_length/2)   
            self.X = pd.concat([ pd.Series(lst), data['aggregate'] , pd.Series(lst)])
            self.y = data[data.columns[-1]]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        # return (torch.tensor(np.array(self.X.iloc[index:index + self.sequence_length])), torch.tensor(np.array(self.y.iloc[[index]])))
        return np.array(self.X.iloc[index:index + self.sequence_length]), np.array(self.y.iloc[[index]])