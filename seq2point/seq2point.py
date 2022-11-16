
import torch
import torch.nn as nn
from pprint import pprint
from torchsummary import summary
from train.training import network_training
from utils.training_utilities import initialize_weights, set_criterion, set_optimization
import os

            
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
            print("\nSEQ2POINT model archiecture has been initialized\n")

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
            print('Saving the model...')
            torch.save(self.state_dict(), os.path.join(TRAINING_CONFIG['SAVE_MODEL'],f'{filename}.pt'))
            
        except Exception as e:
            print("Error occured in save_model method due to ", e)
    
    
    def load_model(self):
        """
        Loads the best model available on the disk location specified in TRAINING_CONFIG.
        """
        try:
            print('Loading the model...')
            self.load_state_dict(torch.load(os.path.join(TRAINING_CONFIG['SAVE_PATH'],TRAINING_CONFIG['LOAD_MODEL'])))
        
        except Exception as e:
            print(f"Error occured in load_model method due to ", e)
            
    
    def run(self, train_loader, validation_loader):
        """
        
        """
        try:
            if TRAINING_CONFIG['PRE_TRAINED_MODEL_FLAG'] == False:
                
                print(f"\nFollowings are the {TRAINING_CONFIG['DESCRIPTION']} of your experiment..")
                pprint(TRAINING_CONFIG) 
                self.apply(initialize_weights) 
                
                print("\nSummary of the model architecture")
                summary(self, (1,599)) ## in progress
                
                criterion = set_criterion()
                optimizer = set_optimization(self)

                train_loss, validation_loss = network_training(self, criterion, optimizer, train_loader, validation_loader)
                return train_loss, validation_loss

            elif TRAINING_CONFIG['PRE_TRAINED_MODEL_FLAG'] == True:
                model.load_model() ## in progress
                summary(model, (1,599)) ## in progress

            else:
                raise Exception('In TRAINING_CONFIG, value specified for PRE_TRAINED_MODEL_FLAG is undefined')
        
        except Exception as e:
            print(f"Error occured in run wrapper method due to ", e)
            

    def inference(test_loader):
        """
        """
        try:
            model.eval()
            start_test_time = time.time()
            test_scores = []
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    data = data.to(set_device)

                    predictions = model.forward(data)

                    ## progress test_scores.append()

            end_test_time = time.time()
            print(f"Testing Loss : {training_loss_per_epoch[-1]}, Time consumption: {end_test_time-start_test_time}s")


        except Exception as e:
                print('Error occured in inference method due to ', e)
            
