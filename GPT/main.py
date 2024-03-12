# import python utils
import random
import time
import re
from tqdm import tqdm

# import ml libraries
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# import self written
from constants import *
from text_dataset import TextDataset
from generate import generate

# notebook i pulled from: https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh?usp=sharing#scrollTo=6ulTWaOr8QNY


######################## SET HYPERPARAMETERS ########################

BATCH_SIZE = 4
EPOCHS = 70
LEARNING_RATE = 3e-5
EPSILON = 1e-8
GAMMA = 0.95
MAX_TOKEN_LENGTH = 768
SAMPLE_RATE = 100

#####################################################################

# set the device to cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# setup model from huggingface
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=False).to(device)



def train():
  # load training and validation data
  training_dataset = TextDataset(INPUT_TRAIN_PATH)
  validation_dataset = TextDataset(INPUT_VAL_PATH)

  # create dataloaders
  training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
  validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

  # setup optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)

  # setup scheduler
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

  # configure model
  model.resize_token_embeddings(len(tokenizer))

  # train model

  model.train()
  for epoch in range(EPOCHS):

    start_time = time.time()
    total_train_loss = 0

    for i, data in tqdm(enumerate(training_dataloader), total=len(training_dataloader)):

      input_ids = data[0].to(device)
      labels = data[0].to(device)
      masks = data[1].to(device)

      model.zero_grad()

      outputs = model(input_ids, attention_mask=masks, labels=labels, token_type_ids=None)

      loss = outputs[0]
      total_train_loss += loss.item()

      if i % SAMPLE_RATE == 0: # sample model
        
        print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
        model.eval()

        sample_outputs = model.generate(
          bos_token_id = random.randint(1,30000), # start with random prompt
          do_sample=True,
          top_k=50,
          max_length = 300,
          top_p=0.95,
          num_return_sequences=1
        )

        for i, sample_output in enumerate(sample_outputs):
          print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

        model.train()

      loss.backward()
      optimizer.step()
      scheduler.step()
      
    print(f"Epoch: {epoch}, Loss: {total_train_loss/len(training_dataloader)}")
    print(f"Time: {time.time() - start_time}")

    ################### validation ###################

    print("Validation:")

    start_time = time.time()
    model.eval()

    total_val_loss = 0

    for batch in validation_dataloader:
      input_ids = batch[0].to(device)
      labels = batch[0].to(device)
      masks = batch[1].to(device)

      with torch.no_grad():
        outputs = model(input_ids, attention_mask=masks, labels=labels, token_type_ids=None)

      loss = outputs[0]
      total_val_loss += loss.item()

    print(f"Epoch: {epoch}, Loss: {total_val_loss/len(validation_dataloader)}")

    # save latest model
        
    model.save_pretrained('gpt2-finetuned')
    tokenizer.save_pretrained('gpt2-finetuned')


if __name__ == "__main__":
  train()
  generate()