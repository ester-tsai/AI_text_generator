import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def generate(model, tokenizer, seq, max_output_length, device=device, temp=0.1, sampling=True, use_topk=True, topk=5):
    model.to(device)
    
    model.eval()
    model.encoder.init_hidden()
    with torch.no_grad():
        if isinstance(seq, str):
            seq = seq.strip().lower()
            tokens = tokenizer(seq, add_special_tokens=True, truncation=True, padding='do_not_pad')['input_ids']
        else:
            tokens = [token for token in seq]
        
        prompt = torch.tensor(tokens).to(device)
        hidden_state = model.encoder(prompt)
        
        predicted = [tokenizer.cls_token_id]
        for _ in range(max_output_length):
            print(predicted)
            inp = torch.tensor(predicted[-1]).unsqueeze(0).to(device)
            out, hidden_state = model.decoder(inp, hidden_state)
            if sampling:
                logits = out / temp
                to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
                logits[to_remove] = -float('inf')
                logits = nn.Softmax(dim=-1)(logits) 
                pred_index = torch.multinomial(logits, 1)[0].item()
                predicted.append(pred_index)
            else:
                predicted.append(out.argmax(-1).item())
            
            # if predict SEP, i.e. end of sentence then break loop
            if predicted[-1] == tokenizer.sep_token_id:
                break
                
    return tokenizer.decode(predicted)