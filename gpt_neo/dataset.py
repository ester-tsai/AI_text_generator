import re
import torch
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
MAX_TOKEN_LENGTH = 256

class TextDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()

        self.dataset_path = dataset_path
        self.input_ids = []
        self.attention_mask = []

        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.data = re.split(r'(?<=<end>)', f.readlines()[0])
            self.data = ['<|startoftext|>' + x[8:-6] + '<|endoftext|>' for x in self.data] # replace <start> and <end> because gpt2 was trained with these tokens
            
            for line in self.data:
                encodings = tokenizer(line, truncation=True, max_length=MAX_TOKEN_LENGTH, padding="max_length")

                self.input_ids.append(torch.tensor(encodings['input_ids']))
                self.attention_mask.append(torch.tensor(encodings['attention_mask'])) # attention mask to ignore padding

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]

    def __len__(self):
        return len(self.input_ids)
