from util import *
from generate import *
from constants import *
from LSTM import *
import torch
from train import *
# from generate import *
import json
import argparse
import gc
import numpy as np
import os

with open(INPUT_TRAIN_PATH, 'r') as train_file:
    with open(INPUT_VAL_PATH, 'r') as val_file:
        token_set = sorted(set(train_file.read().split() + val_file.read().split()))

token_idx_map = {token: index for index, token in enumerate(token_set)}

# TODO determine which device to use (cuda or cpu)
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")   


if __name__ == "__main__":
    #python main.py --config config.json --primer "<start> Ester:" -> To Run the code
    set_seed()

    #Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Specify the config file.')
    parser.add_argument('--primer', type=str, default='<start>', help='The primer string for the generator.')
    args = parser.parse_args()

    
    print("Training on device: ", device)

    # Load the configuration from the specified config file
    with open(f'configs/{args.config}', "r") as config_file:
        config = json.load(config_file)

    # Extract configuration parameters
    max_generation_length = config["max_generation_length"]
    temperature = config["temperature"]   
    learning_rate = config["learning_rate"]
    sequence_size = config["sequence_size"]
    n_sequences = config["n_sequences"]
    n_epochs = config["n_epochs"]
    n_layers = config["n_layers"]
    hidden_size = config["hidden_size"]
    dropout = config["dropout"]
   
    generated_response_folder = config["generated_response_folder"]
    if not os.path.exists(generated_response_folder):
        os.makedirs(generated_response_folder)
    evaluate_model_only = config["evaluate_model_only"]
    model_path = config["model_path"]
    
    info = f'lr{learning_rate}_ep{n_epochs}_temp{temperature}_nseq{n_sequences}_seqsize{sequence_size}_hid{hidden_size}_drop{dropout}_layers{n_layers}'

    # Load training and validation data
    data = load_data(INPUT_TRAIN_PATH, config)
    data_val = load_data(INPUT_VAL_PATH, config)

    print('==> Building model..')

    in_size, out_size = len(token_set), len(token_set)
    # Initialize the LSTM model
    model = LSTM(in_size, out_size, config)

    # If evaluating model only and trained model path is provided:
    if(evaluate_model_only and model_path != ""):
        # Load the checkpoint from the specified model path
        checkpoint = torch.load(model_path)

        # Load the model's state dictionary from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        print('==> Model loaded from checkpoint..')
        
    else:
        # Train the model and get the training and validation losses
        losses, v_losses = train(model, data, data_val, token_idx_map, config, device)
        
#         loss_plot_file_name = f"model={model_type} epochs={n_epochs} layers={n_layers} hidden_size={hidden_size} dropout={dropout} lr={learning_rate} max_len={MAX_GENERATION_LENGTH} temp={TEMPERATURE} seq_size={sequence_size} train_loss={np.round(losses[np.argmin(v_losses)],4)} val_loss={np.round(np.min(v_losses),4)}" 
        
#         # Plot the training and validation losses
#         plot_losses(losses, v_losses, loss_plot_file_name)
    

    # Generate a song from scratch using the trained model
    prime_str = args.primer
    generated_response = generate_response(model, device, token_idx_map, max_len=max_generation_length, temp=temperature, 
                                           prime_str=prime_str)
    
    file_path = os.path.join(generated_response_folder, f'{info} train_loss.txt') 
    with open(file_path, "w+") as file:
        file.write(generated_response)
        

    print("Generated response is written to : ", file_path)

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()


