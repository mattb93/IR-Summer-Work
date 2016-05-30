import os, json

base_dir = "/home/cloudera/Desktop/Spring2016_IR_Project-summer/data"
data_dir = ""
models_dir = os.path.join(base_dir, "models")
predictions_dir = os.path.join(base_dir, "predictions")
FP_dir = os.path.join(base_dir, "FPGrowth")
config_file = os.path.join(base_dir , "collections_config_small.json")

def load_config(config_file):
    """
    Load collection configuration file.
    """
    with open(config_file) as data_file:
        config_data = json.load(data_file)
    return config_data