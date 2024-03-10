from util import *
from constants import *
from LSTM import *
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

def generate_response(model, device, token_idx_map, max_len=100, temp=0.8, prime_str='<start>'):
    """
    Generates a song using the provided model.

    Parameters:
    - model (nn.Module): The trained model used for generating the response
    - device (torch.device): The device (e.g., "cpu" or "cuda") on which the model is located
    - token_idx_map (dict): A map of tokens to their index
    - max_len (int): The maximum length of the generated response
    - temp (float): Temperature parameter for temperature scaling during sampling
    - prime_str (str): Initialize the beginning of the song

    Returns:
    - generated_response (str): The generated song as a string
    """
    # to map index to char for each prediction
    idx_token_map = {v: k for k, v in token_idx_map.items()}
    

    #Move model to the specified device and set the model to evaluation mode
    model.to(device)
    model.eval()

    # Initialize the hidden state
    model.init_hidden()
        
    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
        # "build up" hidden state using the beginging of a response '<start>'
        generated_response = prime_str.split()
        prime = tokens_to_tensor(generated_response, token_idx_map)
        
        # Update hidden state using prime
        for i in range(len(prime)-1):
            inp = prime[i].unsqueeze(0).to(device)
            output, x1 = model(inp)

    
        # Generate new chars
#         while (generated_response[-1] != '<end>') and (len(generated_response) < max_len):
    while len(generated_response) < max_len:
            '''
            TODOs: 
                - Continue generating the rest of the sequence until reaching the maximum length or encountering the end token.
                - Incorporate the temperature parameter to determine the generated/predicted token.
                - Add the generated token to the `generated_response` and then return `generated_response`.
            '''
            inp = tokens_to_tensor(generated_response, token_idx_map)[-1].unsqueeze(0).to(device)
            output, x1 = model(inp)
            softmax_result = nn.Softmax(dim=0)(output[0] / temp)        
            pred_index = torch.multinomial(softmax_result, 1)[0].item()
            pred = idx_token_map[pred_index] 
            generated_response += [pred]
    
    # Turn the model back to training mode
    model.train()

    generated_response = ' '.join(generated_response).replace("<end> <start> ", "\n")
    generated_response = generated_response.replace("<start>", "").replace("<end>", "").strip()
    
    return generated_response
