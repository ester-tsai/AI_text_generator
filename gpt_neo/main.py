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

#inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
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

#train model
model.train()
for epoch in range(epochs):
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids = data[0].to(device)
        masks = data[1].to(device)
        labels = data[0].to(device)
        print(input_ids.shape)
        print(masks.shape)
        print(labels.shape)

        model.zero_grad()

        print(input_ids.device)
        print(masks.device)
        #outputs = model(input_ids=input_ids, attention_mask=masks, labels = labels)
        outputs = model(input_ids, attention_mask=masks, labels=None)
        loss = outputs[0]
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"epoch: {epoch}, loss = {total_train_loss/len(dataloader)}")

