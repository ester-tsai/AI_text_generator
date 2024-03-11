from v1.util import *
from tokenizer import *
import torch.optim as optim
import torch.nn as nn
import numpy as np
import sys
import os


def train_seq2seq(model, tokenizer, train_dataloader, val_dataloader, config, device):

    # Extracting configuration parameters
    n_epochs = config["n_epochs"]
    learning_rate = config["learning_rate"]
    max_sequence_length = config["max_sequence_length"]
    batch_size = config["batch_size"]
    seq_padding = config["seq_padding"]
    pad_index = tokenizer.pad_token_id
    
    info = f'lr{learning_rate}_ep{n_epochs}_seqsize{max_sequence_length}_hid{config["hidden_size"]}_drop{config["dropout"]}_layers{config["n_layers"]}'

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss = nn.CrossEntropyLoss(ignore_index=pad_index) 

    # Lists to store training and validation losses over the epochs
    train_losses, validation_losses = [], []
    min_val_loss = 1e6 # the model should have a val loss lower than this 
    
    # early stopping mechanism
#     loss_increase_epoch_count = 0    
#     prev_val_loss = min_val_loss + 1

    # Training over epochs
    for epoch in range(n_epochs):
        # TRAIN: Train model over training data
        training_loss_per_epoch = []
        val_training_loss_per_epoch = []
        
        train_batch = next(train_dataloader)
        train_prompt = train_batch['prompt']
        train_response = train_batch['response']
        
        model.train()
        
        loss_per_batch = []
        
        count = 0
        
        # do not pad when processing one sequence at a time
        # set padding = 'max_length' if process by batch (not implemented yet)
        for pr, rp in encode_batch(tokenizer, train_prompt, train_response, max_sequence_length, padding=seq_padding):
            
            model.encoder.init_hidden()
            optimizer.zero_grad()
            
            prompt = torch.tensor(pr).to(device)
            response = torch.tensor(rp).to(device)
            
            outputs = model(prompt, response)
            outputs = outputs.to(device)
            
            loss_per_seq = loss(outputs[1:], response[1:])
            loss_per_batch.append(loss_per_seq.item())
            
            loss_per_seq.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            count += 1

            # Display progress
            msg = '\rTraining Epoch: {} iter: {} Loss: {:.4}'.format(epoch, count, loss_per_seq.item())
            sys.stdout.write(msg)
            sys.stdout.flush()
        
        print()
        train_losses.append(np.mean(loss_per_batch))
        

        # VAL: Evaluate Model on Validation dataset
        val_batch = next(val_dataloader)
        val_prompt = val_batch['prompt']
        val_response = val_batch['response']
        
        model.eval() # Put in eval mode (disables batchnorm/dropout) !
        with torch.no_grad(): 
            val_loss_per_batch = []
            count = 0

            for pr, rp in encode_batch(tokenizer, val_prompt, val_response, max_sequence_length, padding=seq_padding):

                prompt = torch.tensor(pr).to(device)
                response = torch.tensor(rp).to(device)

                outputs = model(prompt, response)
                outputs = outputs.to(device)

                val_loss_per_seq = loss(outputs[1:], response[1:])
                val_loss_per_batch.append(val_loss_per_seq.item())

                count += 1

                # Display progress
                msg = '\rValidation Epoch: {} iter: {} Loss: {:.4}'.format(epoch, count, val_loss_per_seq.item())
                sys.stdout.write(msg)
                sys.stdout.flush()

            print()
            
        validation_losses.append(np.mean(val_loss_per_batch))
        print("\n ======= Train best loss: ", min(train_losses), " ======= \n")
        print("\n ======= Validation best loss: ", min(validation_losses), " ======= \n")

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        
        # Save best model
        if validation_losses[-1] < min_val_loss:
            min_val_loss = validation_losses[-1]
            best_epoch = epoch + 1
            best_model = model
            best_optimizer = optimizer
            
#         if mean_val_loss_per_epoch < prev_val_loss:
#             loss_increase_epoch_count = 0
#         else:
#             loss_increase_epoch_count += 1
#         prev_val_loss = mean_val_loss_per_epoch
           
#         if loss_increase_epoch_count >= 3:
#             break
            
            
    print(f'============>Saving best model with min_val_loss={min_val_loss}<=============')
    torch.save({
        'epoch': best_epoch + 1,
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': best_optimizer.state_dict(),
        }, './checkpoint/' + info + f'_best_epoch={best_epoch}_min_train_loss={np.round(train_losses[np.argmin(validation_losses)],4)}_min_val_loss={np.round(min_val_loss, 4)}')
    
    print(f'============>Saving final model<=============')
    torch.save({
        'model_state_dict': model.state_dict(),
        }, './checkpoint/' + info + f'final_model_min_train_loss={min(train_losses)}_min_val_loss={np.round(min_val_loss, 4)}')
            
        
    return train_losses, validation_losses

def generate(model, tokenizer, seq, device, max_output_length):
    model.to(device)
    
    model.eval()
    model.encoder.init_hidden()
    with torch.no_grad():
        if isinstance(seq, str):
            seq = seq.strip().lower()
            tokens = tokenizer(seq, add_special_tokens=True, truncation=True, padding='do_not_pad')['input_ids']
        else:
            tokens = [token for token in seq]
        
        print(tokens)
        prompt = torch.tensor(tokens).to(device)
        hidden_state = model.encoder(prompt)
        print(hidden_state)
        
        predicted = [tokenizer.cls_token_id]
        for _ in range(max_output_length):
            print(predicted)
            inp = torch.tensor(predicted[-1]).unsqueeze(0).to(device)
            out, hidden_state = model.decoder(inp, hidden_state)
            predicted.append(out.argmax(-1).item())
            
            # if predict SEP, i.e. end of sentence then break loop
            if predicted[-1] == tokenizer.sep_token_id:
                break
                
    return tokenizer.decode(predicted)
        
        
        
    