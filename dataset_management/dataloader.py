

import torch
from refit_loader.data_loader import REFIT_Loader
from dataset_management.generator import Sequence2PointGenerator


class Seq2PointDataLoader():
    
    def __init__(self, target_appliance, train_houses, validate_houses, test_houses=[]):
        
        self.__target_appliance = target_appliance
        self.__train_houses = train_houses
        self.__validate_houses = validate_houses
        self.__test_houses = test_houses
        
        self.__appliance_obj = REFIT_Loader().get_appliance_data(appliance=self.__target_appliance,houses=[house for lst_houses in [self.__train_houses,self.__validate_houses,self.__test_houses] for house in lst_houses ])

        self.__appliance_obj.resample(sampling_period = DATASET_CONFIG['SAMPLING_PERIOD'], fill_value = float(DATASET_CONFIG['FILL_VALUE']), window_limit = float(DATASET_CONFIG['WINDOW_LIMIT']) )
        # self.__appliance_obj.dropna()
        
        self.__train_generator = Sequence2PointGenerator(self.__appliance_obj.data[self.__train_houses[0]])
        self.train_dataloader = torch.utils.data.DataLoader(dataset=self.__train_generator, 
                                              batch_size=TRAINING_CONFIG['TRAIN_BATCH_SIZE'], # how many samples per batch
                                              num_workers=0, # how many subprocesses to use for data loading (higher = more)
                                              shuffle=False) # shuffle the data
        
        self.__validation_generator = Sequence2PointGenerator(self.__appliance_obj.data[self.__validate_houses[0]])
        self.validation_dataloader = torch.utils.data.DataLoader(dataset=self.__validation_generator, 
                                              batch_size=TRAINING_CONFIG['VALIDATION_BATCH_SIZE'], # how many samples per batch
                                              num_workers=0, # how many subprocesses to use for data loading (higher = more)
                                              shuffle=False) # shuffle the data    
        if bool(test_houses)==True:
            self.__test_generator = Sequence2PointGenerator(self.__appliance_obj.data[self.__test_houses[0]])
            self.test_dataloader = torch.utils.data.DataLoader(dataset=self.__test_generator, 
                                                  batch_size=TRAINING_CONFIG['TEST_BATCH_SIZE'], # how many samples per batch
                                                  num_workers=0, # how many subprocesses to use for data loading (higher = more)
                                                  shuffle=False) # shuffle the data