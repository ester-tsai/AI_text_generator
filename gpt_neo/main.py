from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from dataset import TextDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

batch_size = 4 
epochs = 5
learning_rate = .01
epsilon = 1e-8
gamma = 0.95
max_token_length = 25


device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")


tokenizer.add_special_tokens({'pad_token': '[PAD]'})
inputs_broken = tokenizer("Hello, my dog is cute. Make this sentence longer and longer to test the functionality" \
    , truncation=True, max_length=max_token_length, padding="max_length", return_tensors="pt")


inputs = tokenizer("Hello, my dog is cute. Make this sentence longer and longer to test the functionality", return_tensors="pt")

print(f'inputs_broken{inputs_broken}')
print(f'inputs_broken_input_ids{inputs_broken["input_ids"].size()}')
print(f'inputs_broken_att_mask{inputs_broken["attention_mask"].size()}')
print(f'inputs{inputs}')

outputs = model(**inputs_broken, labels=inputs["input_ids"])
print(f'outputs1{outputs.logits}')


outputs = model(**inputs, labels=inputs["input_ids"])
print(outputs.logits)

#create dataloader
dataset = TextDataset('../data/cleaned_with_name.txt')
dataloader = DataLoader(dataset, batch_size, shuffle=True)

#set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

#set up scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

#resize token embeddings
model.resize_token_embeddings(len(tokenizer))



#train model
total_train_loss = 0
model.train()
for epoch in range(epochs):
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        print(f'data{data}')
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
        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
        #outputs = model(input_ids=input_ids_test, attention_mask=attention_mask_test)
        #outputs = model(input_ids, attention_mask=masks, labels=None)
        loss = outputs[0]
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"epoch: {epoch}, loss = {total_train_loss/len(dataloader)}")

