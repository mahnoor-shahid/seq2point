
import torch.nn as nn
from pprint import pprint
            
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
            torch.save(self.state_dict(), os.path.join(MODEL_CONFIG['SAVE_PATH'],f'{filename}.pt'))
            
        except Exception as e:
            print("Error occured in save_model method due to ", e)
    
    
    def load_model(self):
        """
        Loads the best model available on the disk location specified in general_config.
        """
        try:
            print('Loading the model...')
            self.load_state_dict(torch.load(os.path.join(MODEL_CONFIG['SAVE_PATH'],MODEL_CONFIG['LOAD_MODEL'])))
        
        except Exception as e:
            print(f"Error occured in load_model method due to ", e)
            
