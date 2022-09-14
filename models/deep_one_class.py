
import torch.nn as nn

class DeepOneClass(nn.Module):

    def __init__(self, model_config: dict):
        """
        This class is created to specify the Seq2Point Network.
        Refer to "ZHANG C, ZHONG M, WANG Z, et al. Sequence-to-point learning with neural networks for non-intrusive load monitoring.
        The 32nd AAAI Conference on Artificial Intelligence"

        Parameters 
        ----------
        model_params : dictionary
            provides the model with the required input channels, output channels, kernel size, stride and padding values
            
            model_params = 
                {
                    'CONV_LAYERS': int,
                    'NODES': list(int),
                    'CONV_KERNEL': list(int),
                    'CONV_STRIDE': int,
                    'CONV_PADDING': int,
                    'POOL_KERNEL' : int,
                    'POOL_STRIDE' : int
                }
        """
        try:
            display("Initializing deep_one_class model archiecture")
            super(deep_one_class, self).__init__()
            
            # assert model_params['SEQUENCE_LENGTH'] >= 599, f"Provided sequence length is {model_params['SEQUENCE_LENGTH']} while it should be atleast >=599"
            self.config = model_config
            self.channels = [i for sublist in [self.config['INPUT_CHANNELS'], self.config['OUTPUT_CHANNELS']] for i in sublist]
            self.layers = []
            
            for layer in range(0, self.config['CONV_LAYERS']):
                self.layers.append(
                    nn.Conv2d(
                        in_channels=self.channels[layer], 
                        out_channels=self.channels[layer+1], 
                        kernel_size=self.config['CONV_KERNEL'][layer],
                        stride=self.config['CONV_STRIDE'], 
                        padding=self.config['CONV_PADDING']))
                self.layers.append(nn.ReLU(inplace=True))
                self.layers.append(nn.MaxPool2d(kernel_size=model_params['CONV_KERNEL'], stride=model_params['CONV_STRIDE']))

            self.features = nn.Sequential(*self.layers)
            self.conv_out = nn.Conv2d(self.channels[-1], num_classes, 2)
            self.softmax = nn.Softmax()

        except Exception:
            print('Error occured in initializing the model architecture due to ', e)

            
    def forward(self, x):
        """
        """
        try:
            num_sam = x.shape[0]
            if len(x.shape) != 4:
                x = x.view(-1,1,28,28)

            feat = self.features(x)
            h = self.conv_out(feat)
            output = h.view(num_sam,-1)
            return output, feat

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
