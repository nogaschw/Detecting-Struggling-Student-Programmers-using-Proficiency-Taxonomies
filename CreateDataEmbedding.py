import gc
import torch
from Data import *
import pandas as pd
from helper import *
from transformers import AutoTokenizer, AutoModel

def process_in_batches(code_model, code_tokenizer, text_list, batch_size, device):
    embedding_dict = {}
    
    for i in range(0, len(text_list), batch_size):
        if i + batch_size < len(text_list):
            batch_text = text_list[i:i + batch_size]
        else:
            batch_text = text_list[i:]
        if i % 1000 == 0:
            print(f"batch {i} / {len(text_list)}", flush=True)

        # Tokenize the batch
        encoding = code_tokenizer(batch_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        with torch.no_grad():
            code_output = code_model(input_ids.to(device), attention_mask.to(device)).last_hidden_state[:, 0, :]

        # Collect rows
        for j, coding in enumerate(batch_text):
            input_ids_str = input_ids[j].cpu().numpy().tobytes()  
            embedding_dict[coding] = code_output[j].tolist()

        # Clear memory
        del input_ids, attention_mask, code_output
        torch.cuda.empty_cache()
        gc.collect()
            
    return embedding_dict



# csv_file_path = '/home/nogaschw/Codeworkout/cleaned_code.csv'
# df = pd.read_csv(csv_file_path, sep=',')
# all_code = set(df[df['course_id'] == 4]['clean_code'].tolist())
df = pd.read_csv('/home/nogaschw/Codeworkout/LinkTables/CodeStates.csv')
all_code = set(df['Code'].tolist())

print(len(all_code))

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

code_model_name = 'microsoft/codebert-base'

code_tokenizer = AutoTokenizer.from_pretrained(code_model_name)
if code_tokenizer.pad_token is None:
    code_tokenizer.pad_token = code_tokenizer.eos_token
code_model = AutoModel.from_pretrained(code_model_name).to(device)

print(f"run with {device}, {code_model_name}")

rows =  process_in_batches(code_model, code_tokenizer, list(all_code), 64, device)
file_path = "/home/nogaschw/Codeworkout/Thesis/Data/codeworkout_to_model_output.pkl"

with open(file_path, 'wb') as f:
    pickle.dump(rows, f)

