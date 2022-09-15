
import torch.nn as nn
            
class SEQ2POINT(nn.Module):

    def __init__(self, model_config: dict):
        """
        This class is created to specify the Seq2Point Network.
        Refer to "ZHANG C, ZHONG M, WANG Z, et al. Sequence-to-point learning with neural networks for non-intrusive load monitoring.
        The 32nd AAAI Conference on Artificial Intelligence"

        Parameters 
        ----------
        model_config : dictionary
            provides the model with the required input channels, output channels, kernel size, stride and padding values
            
            model_config = 
                {
                    'CONV_LAYERS': int,
                    'INPUT_CHANNELS': list(int),
                    'LEFT_PAD': list(int),
                    'RIGHT_PAD': list(int),
                    'OUTPUT_CHANNELS': list(int),
                    'KERNEL': list(int),
                    'STRIDE': int,
                    'PADDING': int,
                    'SEQUENCE_LENGTH': int
                }
        """
        try:
            display("Initializing seq2point model archiecture")
            super(seq2point, self).__init__()
            
            self.config = model_config
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
                        kernel_size=self.config['KERNEL'][layer],
                        stride=self.config['STRIDE'], 
                        padding=self.config['PADDING']))
                self.layers.append(nn.ReLU(inplace=True))
            
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
            raise 

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
            torch.save(self.state_dict(), os.path.join(model_config['SAVE_PATH'],f'{filename}.pt'))
            
        except Exception as e:
            print("Error occured in save_model method due to ", e)
    
    
    def load_model(self):
        """
        Loads the best model available on the disk location specified in general_config.
        """
        try:
            print('Loading the model...')
            self.load_state_dict(torch.load(os.path.join(model_config['SAVE_PATH'],model_config['LOAD_MODEL'])))
        
        except Exception as e:
            print(f"Error occured in load_model method due to ", e)
            
