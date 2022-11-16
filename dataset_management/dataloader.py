

import torch
from refit_loader.data_loader import REFIT_Loader
from dataset_management.generator import Sequence2PointGenerator


class Seq2PointDataLoader():
    """
    This class creates a REFIT_Loader object to load the data for the target_appliance and provided train, validate and test houses
    Further it resamples that data using the SAMPLING_PERIOD, WINDOW_LIMIT and fill additional nans using FILL_VALUE as specified by 'dataset_config.json'
    Then it creates the generator using Sequence2PointGenerator and use that to create pytorch dataloaders and return those created loaders for training, validation and testing
    """
    def __init__(self, target_appliance='kettle', target_houses= dict , proportion= {'train_percent':0.7, 'validate_percent':0.2} , subset_days = None):
        try:
            self.__target_appliance = target_appliance
            self.__target_houses = target_houses
            self.__proportion = proportion
            self.__subset_days = subset_days
            
            if self.__same_house_approach()==True:

                self.__appliance_obj = REFIT_Loader().get_appliance_data(appliance=self.__target_appliance, houses=[self.target_houses['TRAIN']])
                self.__appliance_obj.resample(sampling_period = DATASET_CONFIG['SAMPLING_PERIOD'], fill_value = float(DATASET_CONFIG['FILL_VALUE']), window_limit = float(DATASET_CONFIG['WINDOW_LIMIT']) )

                if bool(self.__subset_days)==True:
                    self.__appliance_obj.subset_data(self.__subset_days)
                    self.__train_df, self.__val_df, self.__test_df = self.__get_proportioned_data(self.__appliance_obj.active_data[self.__same_house])
                else:
                    self.__train_df, self.__val_df, self.__test_df = self.__get_proportioned_data(self.__appliance_obj.data[self.target_houses['TRAIN']])
                                
            else:
                self.__train_houses = self.__target_houses['TRAIN']
                self.__validate_houses = self.__target_houses['VALIDATE']
                self.__test_houses = self.__target_houses['TEST']
                
                self.__appliance_obj = REFIT_Loader().get_appliance_data(appliance=self.__target_appliance, houses=[house for lst_houses in [self.__train_houses,self.__validate_houses,self.__test_houses] for house in lst_houses ])
                self.__appliance_obj.resample(sampling_period = DATASET_CONFIG['SAMPLING_PERIOD'], fill_value = float(DATASET_CONFIG['FILL_VALUE']), window_limit = float(DATASET_CONFIG['WINDOW_LIMIT']) )
                
                if bool(self.__subset_days)==True:
                    self.__appliance_obj.subset_data(self.__subset_days)
                    self.__train_df, self.__val_df, self.__test_df = self.__appliance_obj.active_data[self.__train_houses[0]], self.__appliance_obj.active_data[self.__validate_houses[0]], self.__appliance_obj.active_data[self.__test_houses[0]]
                else:
                    self.__train_df, self.__val_df, self.__test_df = self.__appliance_obj.data[self.__train_houses[0]], self.__appliance_obj.data[self.__validate_houses[0]], self.__appliance_obj.data[self.__test_houses[0]]
        
        except Exception as e:
            print("Error occured in initialization of Seq2PointDataLoader class due to ", e)
            
        finally:
            self.__create_dataloaders()
                
                    
    def __get_proportioned_data(self):
        """
        """
        try:
            self.__train_end = tmp_df.index[math.floor(self.__proportion['train_percent'] * len(self.__tmp_df))]
            self.__val_end = tmp_df.index[math.floor((self.__proportion['train_percent'] + self.__proportion['validate_percent']) * len(self.__tmp_df))]
            return self.self.__tmp_df[:self.__train_end] , self.self.__tmp_df[self.__train_end:self.__val_end], self.self.__tmp_df[self.__val_end:]

        except Exception as e:
            print("Error occured in __get_proportioned_data method due to ", e)
  

    def __same_house_approach(self):
        """
        """
        try:
            if self.target_houses['TRAIN']== self.target_houses['VALIDATE'] and self.target_houses['TRAIN'] == self.target_houses['TEST']:
                return True
            else:
                return False

        except Exception as e:
            print("Error occured in __same_house_approach method due to ", e)
                
                
    def __create_dataloaders(self):
        """
        """
        try:     
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
        except Exception as e:
            print("Error occured in create_dataloaders method due to ", e)