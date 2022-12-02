


import torch
import math
from refit_loader.data_loader import REFIT_Loader
from dataset_management.generator import Sequence2PointGenerator

class Seq2PointDataLoader():
    """
    This class creates a REFIT_Loader object to load the data for the target_appliance and provided train, validate and test houses
    Further it resamples that data using the SAMPLING_PERIOD, WINDOW_LIMIT and fill additional nans using FILL_VALUE as specified by 'dataset_config.json'
    Then it creates the generator using Sequence2PointGenerator and use that to create pytorch dataloaders and return those created loaders for training, validation and testing
    """
    def __init__(self, target_appliance='kettle', target_houses= dict , proportion= dict , subset_days = None, normalize_with = 'standard'):
        try:
            self.__target_appliance = target_appliance
            self.__target_houses = target_houses
            self.__subset_days = subset_days
            self.__normalize_with = normalize_with
            self.__proportion = proportion

            self.__appliance_obj = REFIT_Loader().get_appliance_data(appliance=self.__target_appliance,
                                                                         houses=[house for lst_houses in [self.__target_houses['TRAIN'],
                                                                                                          self.__target_houses['VALIDATE'] ,
                                                                                                          self.__target_houses['TEST']] for house in lst_houses ])

            self.__appliance_obj.resample(sampling_period = DATASET_CONFIG['SAMPLING_PERIOD'], fill_value = float(DATASET_CONFIG['FILL_VALUE']), window_limit = float(DATASET_CONFIG['WINDOW_LIMIT']) )

            if bool(self.__subset_days)==True:
                self.__appliance_obj.subset_data(no_of_days=self.__subset_days)

            self.__appliance_obj.get_proportioned_data(target_houses=self.__target_houses, splits_proportion=self.__proportion)
                
            if bool(self.__normalize_with)==True:
                self.__appliance_obj.normalize(scaler=self.__normalize_with, scalars_directory='scalers/', training = True)

            self.__train_df, self.__val_df, self.__test_df = self.__appliance_obj.splits['TRAIN'], self.__appliance_obj.splits['VALIDATE'], self.__appliance_obj.splits['TEST']
            self.__create_dataloaders()
        
        except Exception as e:
            print("Error occured in initialization of Seq2PointDataLoader class due to ", e)


    def __create_dataloaders(self):
        """
        """
        try:
            print('\nCreating dataloaders...')
            self.__train_generator = Sequence2PointGenerator(self.__train_df)
            self.train_dataloader = torch.utils.data.DataLoader(dataset=self.__train_generator, 
                                                  batch_size=TRAINING_CONFIG['TRAIN_BATCH_SIZE'], # how many samples per batch
                                                  num_workers=0, # how many subprocesses to use for data loading (higher = more)
                                                  shuffle=False) # shuffle the data

            self.__validation_generator = Sequence2PointGenerator(self.__val_df)
            self.validation_dataloader = torch.utils.data.DataLoader(dataset=self.__validation_generator, 
                                                  batch_size=TRAINING_CONFIG['VALIDATION_BATCH_SIZE'], # how many samples per batch
                                                  num_workers=0, # how many subprocesses to use for data loading (higher = more)
                                                  shuffle=False) # shuffle the data    

            self.__test_generator = Sequence2PointGenerator(self.__test_df)
            self.test_dataloader = torch.utils.data.DataLoader(dataset=self.__test_generator, 
                                                  batch_size=TRAINING_CONFIG['TEST_BATCH_SIZE'], # how many samples per batch
                                                  num_workers=0, # how many subprocesses to use for data loading (higher = more)
                                                  shuffle=False) # shuffle the data
            print("Data Loaders are successfully initialized.")
            
        except Exception as e:
            print("Error occured in __create_dataloaders method due to ", e)

            