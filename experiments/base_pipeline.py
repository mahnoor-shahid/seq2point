
class BasePipeline:
    """
    This is the base experimental interface which contains the necessary base functions to be overloaded by any child experimental class
    """

    def __init__(self, model_config: dict):
        print('base')
        self.config = model_config

    def save_model(self):
        """
        Save the best model to the disk location specified in general_config.
        """
        raise NotImplementedError
    
    def load_model(self):
        """
        Loads the best model available on the disk location specified in general_config.
        """
        raise NotImplementedError

    def run(self):
        """
        Wrapper function calling the training and validate functions for every epoch
        """
        raise NotImplementedError

    def train(self):
        """
        Implementation of the training procedure
        """
        raise NotImplementedError

    def validate(self):
        """
        Implementation of the validation procedure
        """
        raise NotImplementedError

    def inference(self):
        """
        Testing of the model
        """
        raise NotImplementedError