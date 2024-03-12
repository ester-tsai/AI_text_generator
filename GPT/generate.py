# ml imports
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate():
    # set the device to cuda if available
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # load model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-finetuned')
    model = GPT2LMHeadModel.from_pretrained('gpt2-finetuned', output_hidden_states=False).to(device)

    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=False).to(device)


    model.eval()
    prompt = "<|startoftext|> Ester:"

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    print(generated)

    sample_outputs = model.generate(
        generated, 
        do_sample=True,   
        top_k=50, 
        max_length = 300,
        top_p=0.95, 
        num_return_sequences=5,
        pad_token_id = tokenizer.eos_token_id
    )

    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
    
if __name__ == "__main__":
    generate()