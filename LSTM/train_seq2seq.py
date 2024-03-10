from util import *
from constants import *
from train import *
from dataLoader import *
import torch.optim as optim
import torch.nn as nn
import numpy as np
import sys
import os


def train_seq2seq(model, data, data_val, token_idx_map, config, device):

    # Extracting configuration parameters
    n_epochs = config["n_epochs"]
    learning_rate = config["learning_rate"]
    hidden_size = config["hidden_size"]
#     dropout = config["dropout"]
    sequence_size = config["sequence_size"]
    batch_size = config["batch_size"]
#     temperature = config["temperature"]
    n_layers = config["n_layers"]
    
    info = f'lr{learning_rate}_ep{n_epochs}_nseq{n_sequences}_seqsize{sequence_size}_hid{hidden_size}_drop{dropout}_layers{n_layers}'

    model = model.to(device) # TODO: Move model to the specified device

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # TODO: Initialize optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss = nn.CrossEntropyLoss() # TODO: Initialize loss function

    # Lists to store training and validation losses over the epochs
    train_losses, validation_losses = [], []
    min_val_loss = 1e6 # the model should have a val loss lower than this 
    loss_increase_epoch_count = 0    
    prev_val_loss = min_val_loss + 1

    # Training over epochs
    for epoch in range(n_epochs):
        # TRAIN: Train model over training data
        training_loss_per_epoch = []
        val_training_loss_per_epoch = []
        for i in range(batch_size):
            '''
            TODO: 
                - For each response:
                    - Zero out/Re-initialise the hidden layer (When you start a new response, the hidden layer state should start at all 0’s.) (Done for you)
                    - Zero out the gradient (Done for you)
                    - Get a random sequence of length: sequence_size from each response (check util.py)
                    - Iterate over sequence tokens : 
                        - Transfer the input and the corresponding ground truth to the same device as the model's
                        - Do a forward pass through the model
                        - Calculate loss per token of sequence
                    - backpropagate the loss after iterating over the sequence of tokens
                    - update the weights after iterating over the sequence of tokens
                    - Calculate avg loss for the sequence
                - Calculate avg loss for the training dataset 
            '''

            model.init_hidden() # Zero out the hidden layer (When you start a new response, the hidden layer state should start at all 0’s.)
            model.zero_grad()   # Zero out the gradient

            #TODO: Finish next steps here
            sequence, target = get_random_response_sequence_target(data, token_idx_map, sequence_size) 
            losses_per_token = [] 
            # Iterate over sequence tokens
            for j in range(len(sequence)):
                inp = torch.tensor(sequence[j]).unsqueeze(0).to(device)
                trg = torch.tensor(target[j]).to(device)
                
                output, _ = model.forward(inp)
                loss_per_token = loss(output.squeeze(0), trg)
                losses_per_token.append(loss_per_token.item())
                
            loss_per_token.backward()
            optimizer.step()
            scheduler.step()

            avg_loss_per_sequence = np.mean(losses_per_token)
            training_loss_per_epoch.append(avg_loss_per_sequence)

            # Display progress
            msg = '\rTraining Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(epoch, (i+1)/n_sequences*100, i, avg_loss_per_sequence)
            sys.stdout.write(msg)
            sys.stdout.flush()

        print()

        # TODO: Append the avg loss on the training dataset to train_losses list
        train_losses.append(np.mean(training_loss_per_epoch))

        # VAL: Evaluate Model on Validation dataset
        model.eval() # Put in eval mode (disables batchnorm/dropout) !
        with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
            # Iterate over validation data
            for i in range(30):
                '''
                TODO: 
                    - For each response:
                        - Zero out/Re-initialise the hidden layer (When you start a new response, the hidden layer state should start at all 0’s.) (Done for you)
                        - Get a random sequence of length: sequence_size from each response
                        - Iterate over sequence tokens : 
                            - Transfer the input and the corresponding ground truth to the same device as the model's
                            - Do a forward pass through the model
                            - Calculate loss per token of sequence
                        - Calculate avg loss for the sequence
                    - Calculate avg loss for the validation dataset 
                '''

                model.init_hidden() # Zero out the hidden layer (When you start a new response, the hidden layer state should start at all 0’s.)

                #TODO: Finish next steps here
                sequence, target = get_random_response_sequence_target(data, token_idx_map, sequence_size)
                
                val_losses_per_token = [] 
                # Iterate over sequence tokens
                for j in range(len(sequence)):
                    inp = torch.tensor(sequence[j]).unsqueeze(0).to(device)
                    trg = torch.tensor(target[j]).to(device)

                    output, _ = model.forward(inp)
                    loss_per_token = loss(output.squeeze(0), trg)
                    val_losses_per_token.append(loss_per_token.item())

                avg_loss_per_sequence = np.mean(val_losses_per_token)
                val_training_loss_per_epoch.append(avg_loss_per_sequence)

                # Display progress
                msg = '\rValidation Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(epoch, (i+1), i, avg_loss_per_sequence)
                sys.stdout.write(msg)
                sys.stdout.flush()

            print()


        # TODO: Append the avg loss on the validation dataset to validation_losses list
        mean_val_loss_per_epoch = np.mean(val_training_loss_per_epoch)
        validation_losses.append(mean_val_loss_per_epoch)
        print("Validation best loss: ", min(validation_losses))

        model.train() #TURNING THE TRAIN MODE BACK ON !

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        
        # Save best model
        if mean_val_loss_per_epoch < min_val_loss:
            min_val_loss = mean_val_loss_per_epoch
            best_epoch = epoch + 1
            best_model = model
            best_optimizer = optimizer
            best_loss = loss
            
       
        if mean_val_loss_per_epoch < prev_val_loss:
            loss_increase_epoch_count = 0
        else:
            loss_increase_epoch_count += 1
        prev_val_loss = mean_val_loss_per_epoch
           
        if loss_increase_epoch_count >= 3:
            break
            
            
    print(f'============>Saving best model with min_val_loss={min_val_loss}<=============')
    torch.save({
#         'epoch': best_epoch + 1,
        'model_state_dict': best_model.state_dict(),
#         'optimizer_state_dict': best_optimizer.state_dict(),
#         'loss': best_loss,
        }, './checkpoint/' + info + f' best_epoch={best_epoch} min_train_loss={np.round(train_losses[np.argmin(validation_losses)],4)} min_val_loss={np.round(min_val_loss, 4)}')
            
        
    return train_losses, validation_losses