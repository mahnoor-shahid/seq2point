
import pandas as pd

def load_dataset():
    """
    """
    try:
        data = pd.read_csv(data_config['DATA_PATH'])
        return data
        
    except Exception as e:
        print("Error occured in load_dataset method due to ", e)