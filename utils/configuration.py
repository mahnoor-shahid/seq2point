
import json

def get_config_from_json(description, config_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    with open(config_file, 'r') as json_file:
        try:
            config_dict = json.load(json_file)
            config_dict['DESCRIPTION']= description
            return config_dict
        
        except ValueError:
            print("Invalid JSON file format. Please provide the correct location of the targeted json file")
            exit(-1)