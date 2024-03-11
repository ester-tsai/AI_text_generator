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
        
        model.encoder.init_hidden()
        optimizer.zero_grad()
        
        loss_per_batch = []
        
        count = 0
        
        for pr, rp in encode_batch(tokenizer, train_prompt, train_response, max_sequence_length):
            
            prompt = torch.tensor(pr).to(device)
            response = torch.tensor(rp).to(device)
            
            outputs = model(prompt, response)
            outputs = outputs.to(device)
            
            loss_per_seq = loss(outputs[1:], response[1:])
            loss_per_batch.append(loss_per_seq.item())
            
            count += 1

            # Display progress
            msg = '\rTraining Epoch: {} iter: {} Loss: {:.4}'.format(epoch, count, loss_per_seq.item())
            sys.stdout.write(msg)
            sys.stdout.flush()
        
        print()
        train_losses.append(np.mean(loss_per_batch))
        loss_per_seq.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # VAL: Evaluate Model on Validation dataset
        val_batch = next(val_dataloader)
        val_prompt = val_batch['prompt']
        val_response = val_batch['response']
        
        model.eval() # Put in eval mode (disables batchnorm/dropout) !
        with torch.no_grad(): 
            val_loss_per_batch = []
            count = 0

            for pr, rp in encode_batch(tokenizer, val_prompt, val_response, max_sequence_length):

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
        print("\n ======= Validation best loss: ", min(validation_losses), " ======= \n")

        model.train() #TURNING THE TRAIN MODE BACK ON !

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
        }, './checkpoint/' + info + f' best_epoch={best_epoch} min_train_loss={np.round(train_losses[np.argmin(validation_losses)],4)} min_val_loss={np.round(min_val_loss, 4)}')
            
        
    return train_losses, validation_losses
    