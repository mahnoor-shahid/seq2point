
import torch
import torch.nn as nn
from train.training import network_training
from utils.training_utilities import initialize_weights, set_criterion, set_optimization
from pprint import pprint
from torchsummary import summary
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
            print(f"Followings are the {MODEL_CONFIG['DESCRIPTION']} of your experiment..")
            pprint(MODEL_CONFIG)
            
            print("\nInitializing SEQ2POINT model archiecture")
            super(SEQ2POINT, self).__init__()
            
            self.config = MODEL_CONFIG
            self.channels = [i for sublist in [self.config['INPUT_CHANNELS'], self.config['OUTPUT_CHANNELS']] for i in sublist]
            
            assert self.config['SEQUENCE_LENGTH'] >= 599, f"Provided sequence length is {self.config['SEQUENCE_LENGTH']} while it should be atleast >=599"
            
            layers = []
            for layer in range(0, self.config['CONV_LAYERS']):
                layers.append(
                    nn.ConstantPad1d(
                        padding=(self.config['LEFT_PAD'][layer], 
                                 self.config['RIGHT_PAD'][layer]), value=0))
                layers.append(
                    nn.Conv1d(
                        in_channels=self.channels[layer], 
                        out_channels=self.channels[layer+1], 
                        kernel_size=self.config['CONV_KERNEL'][layer],
                        stride=self.config['CONV_STRIDE'], 
                        padding=self.config['CONV_PADDING']))
                layers.append(nn.ReLU(inplace=True))
            
            layers.append(
                nn.Linear(
                    in_features=50 * self.config['SEQUENCE_LENGTH'], 
                    out_features=1024))
            layers.append(
                nn.ReLU(inplace=True))
            layers.append(
                nn.Linear(
                    in_features=1024, 
                    out_features=1))

            self.layers = nn.Sequential(*layers)

        except Exception:
            pass
        
        finally:
            print("\nSEQ2POINT model archiecture has been initialized")

    def forward(self, x):
        """
        """
        try:
            return self.layers(x)

        except Exception as e:
            print('Error occured in forward method due to ', e)
 

    def save_model(self, filename):
        """
        Save the best model to the disk location specified in general_config.
        
        Parameter
        ----------
        file_name : string
            Name of the file of the saved model 
        """
        try:
            print('Saving the model...')
            print(GENERAL_CONFIG['SAVE_PATH'] )

            # Check whether the specified path exists or not
            if not os.path.exists(GENERAL_CONFIG['SAVE_PATH'] ):
                print("no path")
                os.makedirs(GENERAL_CONFIG['SAVE_PATH'] ) 
            torch.save(self.state_dict(), os.path.join(GENERAL_CONFIG['SAVE_PATH'],f'{filename}.pt'))
            
        except Exception as e:
            print("Error occured in save_model method due to ", e)
    
    
    def load_model(self):
        """
        Loads the best model available on the disk location specified in general_config.
        """
        try:
            print('Loading the model...')
            self.load_state_dict(torch.load(os.path.join(GENERAL_CONFIG['SAVE_PATH'],GENERAL_CONFIG['LOAD_MODEL'])))
        
        except Exception as e:
            print(f"Error occured in load_model method due to ", e)
            
    
    def run(self):
        """
        
        """
        try:
            if GENERAL_CONFIG['PRE_TRAINED_MODEL_FLAG'] == False:
                
                print(f"Followings are the {TRAINING_CONFIG['DESCRIPTION']} of your experiment..")
                pprint(TRAINING_CONFIG) 
                self.apply(initialize_weights) 
                
                print("\nSummary of the model architecture")
                summary(self, (1,599)) ## in progress
                
                criterion = set_criterion()
                optimizer = set_optimization(self)
                train_loader = torch.randn(5,2)
                validation_loader = torch.randn(2,2)

                train_loss, validation_loss = network_training(self, criterion, optimizer, train_loader, validation_loader)
                return train_loss, validation_loss

            elif GENERAL_CONFIG['PRE_TRAINED_MODEL_FLAG'] == True:
                model.load_model() ## in progress
                summary(model, (1,599)) ## in progress

            else:
                raise Exception('In general_config, value specified for PRE_TRAINED_MODEL_FLAG is undefined')
        
        except Exception as e:
            print(f"Error occured in run wrapper method due to ", e)
            

    def inference():
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
            
