from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from dataset import TextDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
print(f'inputs{inputs}')
#outputs = model(**inputs, labels=inputs["input_ids"])
#print(outputs.logits)

#prompt = (
#    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
#    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
#    "researchers was the fact that the unicorns spoke perfect English."
#)

#input_ids = tokenizer(prompt, return_tensors="pt").input_ids

#gen_tokens = model.generate(
#    input_ids,
#    do_sample=True,
#    temperature=0.9,
#    max_length=100,
#)
#gen_text = tokenizer.batch_decode(gen_tokens)[0]

batch_size = 4 
epochs = 5
learning_rate = .01
epsilon = 1e-8
gamma = 0.95
max_token_length = 100

#create dataloader
dataset = TextDataset('../data/cleaned_with_name.txt')
dataloader = DataLoader(dataset, batch_size, shuffle=True)

#set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

#set up scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

#resize token embeddings
model.resize_token_embeddings(len(tokenizer))


input_ids_test = [   27,    91,  9688,  1659,  5239,    91,    29, 17100,    25, 19462,
           262, 13439,   836,   447,   247,    83,   787,  2565, 30325,   255,
         50256, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257,
         50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257,
         50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257,
         50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257,
         50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257,
         50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257,
         50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257,
         50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257]

attention_mask_test = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0] 

inputs_test = {'input_ids': torch.tensor(input_ids_test), 'attention_mask': torch.tensor(attention_mask_test)}
#inputs = {'input_ids': tensor([[15496,    11,   616,  3290,   318, 13779]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}

#train model
model.train()
for epoch in range(epochs):
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids = data[0].to(device)
        attention_mask = data[1].to(device)
        labels = data[0].to(device)
        print(f'input_ids.shape{input_ids.shape}')
        print(f'attention_mask.shape{attention_mask.shape}')
        print(f'labels.shape{labels.shape}')
        print(f'input_ids{input_ids}')
        print(f'attention_mask{attention_mask}')

        model.zero_grad()

        print(input_ids.device)
        print(attention_mask.device)
        outputs = model(**inputs_test, labels=input_ids)
        #outputs = model(input_ids=input_ids_test, attention_mask=attention_mask_test)
        #outputs = model(input_ids, attention_mask=masks, labels=None)
        loss = outputs[0]
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"epoch: {epoch}, loss = {total_train_loss/len(dataloader)}")

