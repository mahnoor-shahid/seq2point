
import torch
import torch.nn as nn
from pprint import pprint
from torchsummary import summary
import datetime
from training.train import network_train, fetch_training_reports
from utils.training_utilities import set_GPU, initialize_weights, set_criterion, set_optimization
import os
import json
import numpy as np

            
class SEQ2POINT(nn.Module):

    def __init__(self):
        """
        This class is created to specify the Seq2Point Network.
        Refer to "ZHANG C, ZHONG M, WANG Z, et al. Sequence-to-point learning with neural networks for non-intrusive load monitoring.
        The 32nd AAAI Conference on Artificial Intelligence"

        Parameters 
        ----------
        MODEL_CONFIG : dictionary
            provides the model with the required input channels, output channels, kernel size, stride and padding values
            
            MODEL_CONFIG = 
                {
                    'CONV_LAYERS': int,
                    'INPUT_CHANNELS': list(int),
                    'LEFT_PAD': list(int),
                    'RIGHT_PAD': list(int),
                    'OUTPUT_CHANNELS': list(int),
                    'CONV_KERNEL': list(int),
                    'CONV_STRIDE': int,
                    'CONV_PADDING': int,
                    'SEQUENCE_LENGTH': int
                }
        """
        try:
            print("\nInitializing SEQ2POINT model archiecture\n")
            print(f"Followings are the {MODEL_CONFIG['DESCRIPTION']} of your network architecture..")
            pprint(MODEL_CONFIG)
            
            super(SEQ2POINT, self).__init__()
            self.__config = MODEL_CONFIG
            
            conv_layers = []
            dense_layers = []
            for layer in range(0, self.__config['CONV_LAYERS']):
                conv_layers.append(
                    nn.ConstantPad1d(
                        padding=(self.__config['LEFT_PAD'][layer], 
                                 self.__config['RIGHT_PAD'][layer]), value=0))
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=self.__config['INPUT_CHANNELS'][layer], 
                        out_channels=self.__config['OUTPUT_CHANNELS'][layer], 
                        kernel_size=self.__config['CONV_KERNEL'][layer],
                        stride=self.__config['CONV_STRIDE'], 
                        padding=self.__config['CONV_PADDING']))
                conv_layers.append(nn.ReLU(inplace=True))
            self.conv = nn.Sequential(*conv_layers)
            
            dense_layers.append(
                nn.Linear(
                    in_features=self.__config['LINEAR_INPUT'][0], 
                    out_features=self.__config['LINEAR_OUTPUT'][0]))
            dense_layers.append(
                nn.ReLU(inplace=True))
            dense_layers.append(
                nn.Linear(
                    in_features=self.__config['LINEAR_INPUT'][1], 
                    out_features=self.__config['LINEAR_OUTPUT'][1]))
            self.dense = nn.Sequential(*dense_layers)

        except Exception:
            pass
        
        finally:
            print("\nSEQ2POINT model archiecture has been initialized.\n")

    def forward(self, x):
        """
        """
        try:
            x = self.conv(x)
            x = self.dense(x.view(-1, 50 * self.__config['SEQUENCE_LENGTH']))
            return x

        except Exception as e:
            print('Error occured in forward method due to ', e)
 

    def save_model(self, filename):
        """
        Save the best model to the disk location specified in TRAINING_CONFIG.
        
        Parameter
        ----------
        file_name : string
            Name of the file of the saved model 
        """
        try:
            print(f"Saving the {filename} model...\n")
            if not os.path.exists(os.path.join(TRAINING_CONFIG['EXPERIMENT_PATH'], 'models')):
                os.makedirs(os.path.join(TRAINING_CONFIG['EXPERIMENT_PATH'], 'models'))

            TRAINING_CONFIG['BEST_MODEL'] = os.path.join(TRAINING_CONFIG['EXPERIMENT_PATH'], f'models/{filename}.pt')
            torch.save(self.state_dict(), TRAINING_CONFIG['BEST_MODEL'])
            
        except Exception as e:
            print("Error occured in save_model method due to ", e)
    
    
    def load_model(self):
        """
        Loads the best model available on the disk location specified in TRAINING_CONFIG.
        """
        try:
            print(f"Loading the model...{TRAINING_CONFIG['BEST_MODEL']}")
            self.load_state_dict(torch.load(TRAINING_CONFIG['BEST_MODEL']))
        
        except Exception as e:
            print(f"Error occured in load_model method due to ", e)
            
    
    def run(self, train_loader, validation_loader, assess_training):
        """
        
        """
        try:
            if TRAINING_CONFIG['PRE_TRAINED_MODEL_FLAG'] == False:

                if not os.path.exists(TRAINING_CONFIG['EXPERIMENT_PATH']):
                    os.makedirs(TRAINING_CONFIG['EXPERIMENT_PATH'])
                with open(os.path.join(TRAINING_CONFIG['EXPERIMENT_PATH'], 'experiment_config.json'), 'w') as json_file:
                    json.dump(TRAINING_CONFIG, json_file)
                self.apply(initialize_weights)

                print(f"\nFollowings are the {TRAINING_CONFIG['DESCRIPTION']} of your experiment..")
                pprint(TRAINING_CONFIG)
                
                print("\nSummary of the model architecture")
                summary(self, (1,599)) ## in progress
                
                criterion = set_criterion()
                optimizer = set_optimization(self)

                results = network_train(self, criterion, optimizer, train_loader, validation_loader, assess_training)
                return results

            elif TRAINING_CONFIG['PRE_TRAINED_MODEL_FLAG'] == True:
                model.load_model() ## in progress
                summary(model, (1,599)) ## in progress

            else:
                raise Exception('In TRAINING_CONFIG, value specified for PRE_TRAINED_MODEL_FLAG is undefined')
        
        except Exception as e:
            print(f"Error occured in run wrapper method due to ", e)
            

    def inference(self, test_loader):
        """
        """
        try:
            self.load_model()

            print("Model's state_dict:")
            for param_tensor in self.state_dict():
                print(param_tensor, "\t", self.state_dict()[param_tensor].size())
    
            criterion = set_criterion()
            
            start_test_time = datetime.datetime.now()
            test_scores = []
            self.eval()
            with torch.no_grad():
                for batch_idx, (timestep, x_value, y_value) in enumerate(test_loader):
                    timestep = [datetime.datetime.fromtimestamp(each_timestep).strftime('%Y-%m-%d %H:%M:%S') for each_timestep in timestep.numpy()]
                    x_value = x_value[:, None].type(torch.cuda.FloatTensor).to(set_GPU())
                    y_value = y_value[:, None].type(torch.cuda.FloatTensor).to(set_GPU())
                    predictions = self.forward(x_value)[:, None].type(torch.cuda.FloatTensor).to(set_GPU())             
                    
                    loss = criterion(y_value, predictions)

                    test_scores.append(loss.item())

            end_test_time = datetime.datetime.now()
            print(f"Average Test Loss : {np.mean(test_scores)}, Time consumption: {end_test_time-start_test_time}s")
            
            return test_scores


        except Exception as e:
                print('Error occured in inference method due to ', e)
            

#%%
