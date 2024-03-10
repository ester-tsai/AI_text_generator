import random
import torch
from constants import *
import matplotlib.pyplot as plt
import os
import numpy as np
import json

def set_seed(seed: int = 10):
    pass
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
# #     torch.use_deterministic_algorithms(True, warn_only=True)
#     # When running on the CuDNN backend, two further options must be set
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # Set a fixed value for the hash seed
#     os.environ["PYTHONHASHSEED"] = str(seed)

def load_config(fp):
    """
    Loads config JSON file

    Args:
        fp (string): The filepath of config.json

    Returns:
        dict: A dict of config
    """
    with open(fp) as f:
        config = json.load(f)
    return config


def get_random_text_slice(data, sequence_length):
    """
    TODO: Retrieves a random slice of the given data with the specified sequence length.

    Args:
        data (str): The input data (e.g., text notation).
        sequence_length (int): The desired length of the sequence to extract.

    Returns:
        list: A random slice of the input data with the specified sequence length.
    """
    
    max_start_index = max(0, len(data)-sequence_length)

    random_start_index = random.randint(0, max_start_index)
    while data[random_start_index] != '<start>':
        random_start_index -= 1
    sliced_text = data[random_start_index: random_start_index+sequence_length]

    assert len(sliced_text) <= sequence_length

    return sliced_text
    
def tokens_to_tensor(sequence, token_idx_map):
    """
    Converts a sequence of tokens to a PyTorch tensor using the provided token set.
    (DON'T CHANGE)

    Args:
        sequence (list): The sequence of tokens to convert.
        token_idx_map (dict): A map of tokens to their index

    Returns:
        torch.Tensor: A PyTorch tensor representing the input sequence.
    """
    return torch.tensor([token_idx_map[token] for token in sequence], dtype=torch.long)

def get_random_response_sequence_target(text, token_idx_map, sequence_length):
    """
    Retrieves a random sequence from the given text data along with its target sequence.
    (DON'T CHANGE)

    Args:
        text (list): The text data, represented as a list of tokens.
        token_idx_map (dict): A map of tokens to their index
        sequence_length (int): The desired length of the sequence to extract.

    Returns:
        tuple: A tuple containing the PyTorch tensor representing the input sequence 
               and the PyTorch tensor representing the target sequence.
    """
    sequence = get_random_text_slice(text, sequence_length)
    sequence_tensor = tokens_to_tensor(sequence[:-1], token_idx_map)
    target_tensor = tokens_to_tensor(sequence[1:], token_idx_map)
    return sequence_tensor, target_tensor


def get_token_from_index(token_idx_map, index):
    """
    (DON'T CHANGE)
    """
    for token, idx in token_idx_map.items():
        if idx == index:
            return token
    # If value is not found, return None or raise an error as needed
    return None






def plot_losses(train_losses, val_losses, fname):
    """
    Plots the training and validation losses across epochs and saves the plot as an image file with name - fname(function argument). 

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        fname (str): Name of the file to save the plot (without extension).

    Returns:
        None
    """

    # Create 'plots' directory if it doesn't exist

    if not os.path.isdir('plots'):
        os.mkdir('plots')

    # Plotting training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/" + fname + ".png")


