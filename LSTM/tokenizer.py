from constants import *
import json
import gc
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast, BertTokenizerFast

def create_tokenizer(text_fp, tokenizer_save_fp='tokenizer/', tokenizer_name='bert-base-uncased', max_vocab_size=32000):
    lines = []
    with open(text_fp) as file:
        for line in file: 
            line = line.strip() 
            lines.append(line)
            
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = tokenizer.train_new_from_iterator(iter(lines), max_vocab_size)
    print("Vocab size is: ", tokenizer.vocab_size)
    tokenizer.save_pretrained(tokenizer_save_fp)
    
def load_tokenizer(vocab_fp='tokenizer/vocab.txt'):
    tokenizer = BertTokenizerFast(vocab_fp)
    return tokenizer