import sys
print(sys.path)

from tokenizer import *
from transformers import set_seed
from v1.dataLoader import *
from LSTM_Seq2Seq import *
import torch
from train_seq2seq import *
import json
import argparse
import gc
import numpy as np
import os



device = 'cuda' if torch.cuda.is_available() else 'cpu' 

if __name__ == "__main__":
    #python3 main.py --config config.json  -> To Run the code
    set_seed(100)

    #Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Specify the config file.')
    args = parser.parse_args()

    
    print("Training on device: ", device)

    # Load the configuration from the specified config file
    with open(f'configs/{args.config}', "r") as config_file:
        config = json.load(config_file)

    # Extract configuration parameters
    max_sequence_length = config["max_sequence_length"]
    temperature = config["temperature"]   
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    n_epochs = config["n_epochs"]
    n_layers = config["n_layers"]
    hidden_size = config["hidden_size"]
    dropout = config["dropout"]
   
    generated_response_folder = config["generated_response_folder"]
    if not os.path.exists(generated_response_folder):
        os.makedirs(generated_response_folder)
    evaluate_model_only = config["evaluate_model_only"]
    model_path = config["model_path"]
    
    info = f'lr{learning_rate}_ep{n_epochs}_temp{temperature}_bs{batch_size}_hid{hidden_size}_drop{dropout}_layers{n_layers}'

    # Load training and validation data
    print('==> Loading train/val data..')
    
    train = load_jsonl('../data/prompt_response/train.jsonl')
    train_dl = create_data_loader(train, batch_size=batch_size)
    
    val = load_jsonl('../data/prompt_response/valid.jsonl')
    val_dl = create_data_loader(val)
    
    # Load pre-trained tokenizer
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.vocab_size
    print('==> Vocab size is: ', vocab_size)

    print('==> Building model..')
    # Initialize encoder-decoder model
    encoder = Encoder(vocab_size, config)
    decoder = Decoder(vocab_size, config)
    model = Seq2Seq(encoder, decoder)
    

    # If evaluating model only and trained model path is provided:
    if(evaluate_model_only and model_path != ""):
        # Load the checkpoint from the specified model path
        checkpoint = torch.load(model_path)

        # Load the model's state dictionary from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        print('==> Model loaded from checkpoint..')
        
    else:
        # Train the model and get the training and validation losses
        losses = train_seq2seq(model, tokenizer, train_dl, val_dl, config, device)
        
#         loss_plot_file_name = f"model={model_type} epochs={n_epochs} layers={n_layers} hidden_size={hidden_size} dropout={dropout} lr={learning_rate} max_len={MAX_GENERATION_LENGTH} temp={TEMPERATURE} seq_size={sequence_size} train_loss={np.round(losses[np.argmin(v_losses)],4)} val_loss={np.round(np.min(v_losses),4)}" 
        
#         # Plot the training and validation losses
#         plot_losses(losses, v_losses, loss_plot_file_name)
    

    # Generate a song from scratch using the trained model
#     prime_str = args.primer
#     generated_response = generate_response(model, device, token_idx_map, max_len=max_generation_length, temp=temperature, 
#                                            prime_str=prime_str)
    
#     file_path = os.path.join(generated_response_folder, f'{prime_str} {info}.txt') 
#     with open(file_path, "w+") as file:
#         file.write(generated_response)
        

#     print("Generated response is written to : ", file_path)

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()


