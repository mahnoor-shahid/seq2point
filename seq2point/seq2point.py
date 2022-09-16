

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
            print("\nInitializing SEQ2POINT model archiecture\n")
            print(f"Followings are the {MODEL_CONFIG['DESCRIPTION']} of your network architecture..")
            pprint(MODEL_CONFIG)
            
            super(SEQ2POINT, self).__init__()
            self.config = MODEL_CONFIG
            
            assert self.config['SEQUENCE_LENGTH'] >= 599, f"Provided sequence length is {self.config['SEQUENCE_LENGTH']} while it should be atleast >=599"
            
            conv_layers = []
            dense_layers = []
            for layer in range(0, self.config['CONV_LAYERS']):
                conv_layers.append(
                    nn.ConstantPad1d(
                        padding=(self.config['LEFT_PAD'][layer], 
                                 self.config['RIGHT_PAD'][layer]), value=0))
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=self.config['INPUT_CHANNELS'][layer], 
                        out_channels=self.config['OUTPUT_CHANNELS'][layer], 
                        kernel_size=self.config['CONV_KERNEL'][layer],
                        stride=self.config['CONV_STRIDE'], 
                        padding=self.config['CONV_PADDING']))
                conv_layers.append(nn.ReLU(inplace=True))
            self.conv = nn.Sequential(*conv_layers)
            
            dense_layers.append(
                nn.Linear(
                    in_features=self.config['LINEAR_INPUT'][0], 
                    out_features=self.config['LINEAR_OUTPUT'][0]))
            dense_layers.append(
                nn.ReLU(inplace=True))
            dense_layers.append(
                nn.Linear(
                    in_features=self.config['LINEAR_INPUT'][1], 
                    out_features=self.config['LINEAR_OUTPUT'][1]))
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
            x = self.dense(x.view(-1, 50 * self.config['SEQUENCE_LENGTH']))
            return x.view(-1, 1)

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
            
    
    def run(self, train_loader, validation_loader):
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
            
