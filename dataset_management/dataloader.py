

import torch
from refit_loader.data_loader import REFIT_Loader
from dataset_management.generator import Sequence2PointGenerator


class Seq2PointDataLoader():
    """
    This class creates a REFIT_Loader object to load the data for the target_appliance and provided train, validate and test houses
    Further it resamples that data using the SAMPLING_PERIOD, WINDOW_LIMIT and fill additional nans using FILL_VALUE as specified by 'dataset_config.json'
    Then it creates the generator using Sequence2PointGenerator and use that to create pytorch dataloaders and return those created loaders for training, validation and testing
    """
    def __init__(self, target_appliance: str, train_houses: list, validate_houses: list, test_houses: list, subset_days: int):
        try:
            pass
        
        except Exception as e:
            print("Error occured in initialization of Seq2PointDataLoader class due to ", e)
            
        finally:
            self.__target_appliance = target_appliance
            self.__train_houses = train_houses
            self.__validate_houses = validate_houses
            self.__test_houses = test_houses

            self.__appliance_obj = REFIT_Loader().get_appliance_data(appliance=self.__target_appliance,houses=[house for lst_houses in [self.__train_houses,self.__validate_houses,self.__test_houses] for house in lst_houses ])

            self.__appliance_obj.resample(sampling_period = DATASET_CONFIG['SAMPLING_PERIOD'], fill_value = float(DATASET_CONFIG['FILL_VALUE']), window_limit = float(DATASET_CONFIG['WINDOW_LIMIT']) )
            
            if bool(subset_days)==True:
                print(f'Making a subset for {subset_days} days of active data')
                self.__appliance_obj.subset_data(subset_days)
                self.__train_generator = Sequence2PointGenerator(self.__appliance_obj.active_data[self.__train_houses[0]])
                self.train_dataloader = torch.utils.data.DataLoader(dataset=self.__train_generator, 
                                                      batch_size=TRAINING_CONFIG['TRAIN_BATCH_SIZE'], # how many samples per batch
                                                      num_workers=0, # how many subprocesses to use for data loading (higher = more)
                                                      shuffle=False) # shuffle the data

                self.__validation_generator = Sequence2PointGenerator(self.__appliance_obj.active_data[self.__validate_houses[0]])
                self.validation_dataloader = torch.utils.data.DataLoader(dataset=self.__validation_generator, 
                                                      batch_size=TRAINING_CONFIG['VALIDATION_BATCH_SIZE'], # how many samples per batch
                                                      num_workers=0, # how many subprocesses to use for data loading (higher = more)
                                                      shuffle=False) # shuffle the data    
                if bool(test_houses)==True:
                    self.__test_generator = Sequence2PointGenerator(self.__appliance_obj.active_data[self.__test_houses[0]])
                    self.test_dataloader = torch.utils.data.DataLoader(dataset=self.__test_generator, 
                                                          batch_size=TRAINING_CONFIG['TEST_BATCH_SIZE'], # how many samples per batch
                                                          num_workers=0, # how many subprocesses to use for data loading (higher = more)
                                                          shuffle=False) # shuffle the data
            else:
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