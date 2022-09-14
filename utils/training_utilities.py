
import torch

def initialize_weights(layer): #torch.nn.modules
    """
    Initializing weights using Xavier Initialization for every Conv1D and Linear Layer
        
    Parameters
    ----------
    layer : torch.nn.modules
    
    """
    try:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight.data)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias.data, val=0.0)     
            
    except Exception as e:
        print("Error occured in initialize_weights method due to ", e)

        
        
def set_optimization(model):
    """
    """
    
    try:
        return training_config['OPTIMIZER'](model.parameters(), lr=training_config['LEARNING_RATE'])
    
    except Exception as e:
        print("Error occured in set_optimization method due to ", e)


        
def set_criterion():
    """
    """
    
    try:
        return training_config['LOSS'](reduction=training_config['LOSS_REDUCTION'])
    
    except Exception as e:
        print("Error occured in set_criterion method due to ", e)
     
    
    
def set_GPU():
    """
    """
    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    except Exception as e:
        print("Error occured in set_GPU_flag method due to ", e)

        
        
def early_stopping(idle_training_epochs):
    """
    """
    try:
        if idle_training_epochs == training_config['EARLY_STOPPING_THRESHOLD']:
            print("Earlystopping is calling it off because validation loss did not improve after {} epochs, therefore e".format(idle_training_epochs))
            return True 
    
    except Exception as e:
        print("Error occured in early_stopping method due to ", e)
        