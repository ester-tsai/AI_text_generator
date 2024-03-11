from v1.constants import *
import json
import gc
import os
from transformers import AutoTokenizer, BertTokenizerFast

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

def encode_prompt_response(tokenizer, prompt, response, max_length, add_special_tokens=True, truncation=True, padding="max_length"):
    tokenized_prompt = tokenizer(prompt, add_special_tokens=add_special_tokens, truncation=truncation, max_length=max_length, padding=padding)['input_ids']
    tokenized_response = tokenizer(response, add_special_tokens=add_special_tokens, truncation=truncation, max_length=max_length, padding=padding)['input_ids']
    
    return tokenized_prompt, tokenized_response

def encode_batch(tokenizer, prompts, responses, max_length, add_special_tokens=True, truncation=True, padding="max_length"):
    
    for prompt, response in zip(prompts, responses):
        tokenized_prompt, tokenized_response = encode_prompt_response(tokenizer, prompt, response, add_special_tokens=add_special_tokens, truncation=truncation, max_length=max_length, padding=padding)
        yield tokenized_prompt, tokenized_response
        
    