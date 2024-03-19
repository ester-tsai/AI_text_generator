from torch.utils.data import DataLoader
import json
import random

def load_jsonl(fp):
    data = []
    with open(fp, 'r') as json_file:
        for row in json_file:
            content = json.loads(row)
            data.append(content)
    return data
    
def load_prompt_reponse(file, cutoff_percentage=0.8, test_count=10):
    sentence_lst = []
    with open(file, 'r') as json_file:
        for row in json_file:
            content = json.loads(row)
            sentence_lst.append(content)
    train, val, test = generate_train_val_test(sentence_lst, cutoff_percentage=cutoff_percentage, test_count=test_cont)
    
    return train, val, test
    
    
def load_data(file, config):
    """
    Load messages from a file.

    Parameters:
    - file (str): The path to the data file
    - config (dict): A dictionary containing configuration parameters:
        - "sequence_size" (int): The size of the sequences to extract from the data and train our model. 

    Returns:
    - data (list): A list of sequences extracted from the data file
    """

    # Extract configuration parameters
    SEQ_SIZE = config["sequence_size"]

    # Initialize an empty list to store the data
    data = []

    # Read data from the file
    with open(file, "r") as f:
        string_of_all_texts = f.readlines()[0]
        return string_of_all_texts.split()
    

def generate_train_val_test(data, cutoff_percentage=0.8, test_count=10):
    random.shuffle(data) 
    n = len(data)
    train_cutoff = int(n*cutoff_percentage)
    train_data = data[:train_cutoff]
    val_data = data[train_cutoff:-test_count]
    test_data = data[-test_count:]
    
    return train_data, val_data, test_data

def create_data_loader(data, batch_size=10, loop=True, shuffle=True):

    data_loader = DataLoader(
                data,
                batch_size=batch_size,
                shuffle=shuffle)

    if loop:
        return infinite_loader(data_loader)
    else:
        # print(data_loader)
        return iter(data_loader)

def infinite_loader(data_loader):
    while True:
        yield from data_loader