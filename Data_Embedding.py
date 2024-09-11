import torch
import pickle
from torch.utils.data import Dataset

with open('/home/nogaschw/Codeworkout/Thesis/Data/merged_dict.pkl', 'rb') as file:
    code_to_model_dict = pickle.load(file)

class Dataset_Embedding(Dataset):
    def __init__(self, df, text_tokenizer, code_tokenizer, max_len_text=512, max_len_code=768, limit_len=0, padding_size_code=2000, padding_size_q=200, feature=None):
        self.df = df
        self.text_tokenizer = text_tokenizer
        self.max_len_text = max_len_text
        self.max_len_code = max_len_code
        self.limit_len = limit_len
        self.padding_size_code = padding_size_code
        self.padding_size_q = padding_size_q
        self.feature_columns = feature

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['question']
        code_samples = row['prev_code']
        prev_question = row['prev_question']
        label = torch.tensor(row['struggling'], dtype=torch.float)
    
        limit_len = self.padding_size_q if self.limit_len == 0 else min(self.limit_len, self.padding_size_q)

        if limit_len: # diffrent from 0
            if len(code_samples) > limit_len:
                code_samples = code_samples[-limit_len:]
                prev_question = prev_question[-limit_len:]
        
        # Tokenize curr coding text
        text_inputs = self.text_tokenizer(question, max_length=self.max_len_text, padding='max_length', truncation=True, return_tensors='pt')
        
        snapshots = []
        len_snapshots = [0 for i in range(self.padding_size_q)]
        for idx, codes in enumerate(code_samples):
            len_snapshots[idx] = len(codes)
            snapshots.extend(codes)

        # embedded for codes
        embedding = torch.zeros((self.padding_size_code, self.max_len_code), dtype=torch.long)
        for idx, code in enumerate(snapshots):
            embedding[idx, :] = torch.tensor(code_to_model_dict[code])
        
        # Tokenize per question
        if len(prev_question) < self.padding_size_q:
            pad_size = torch.zeros((self.padding_size_q - len(prev_question), self.max_len_text), dtype=torch.long)
        else:
            pad_size = torch.zeros((0, self.max_len_text), dtype=torch.long)
        inputs = [self.text_tokenizer(sample, max_length=self.max_len_text, padding='max_length', truncation=True, return_tensors='pt') for sample in prev_question]
        question_input_ids_stack = torch.cat([torch.stack([data['input_ids'] for data in inputs]).squeeze(1),pad_size], dim=0)
        question_attention_mask_stack = torch.cat([torch.stack([data['attention_mask'] for data in inputs]).squeeze(1), pad_size], dim=0)

        return {
            'text_input_ids': text_inputs['input_ids'].squeeze(0),
            'text_attention_mask': text_inputs['attention_mask'].squeeze(0),
            'code_embedding': embedding,
            'code_num': len(snapshots),
            'questions_input_ids': question_input_ids_stack.squeeze(0),
            'questions_attention_mask': question_attention_mask_stack.squeeze(0),
            'num_snapshots': torch.tensor(len_snapshots),
            'que_num': len(prev_question),
            'label': label
        }
    
class FeatureDataset_Embedding(Dataset_Embedding):
    def __init__(self, df, text_tokenizer, code_tokenizer, max_len_text=512, max_len_code=768, limit_len=0, padding_size_code=2000, padding_size_q=200 ,feature=None):
        super().__init__(df, text_tokenizer, code_tokenizer, max_len_text=max_len_text, max_len_code=max_len_code, padding_size_code=padding_size_code, padding_size_q=padding_size_q, limit_len=limit_len)
        self.feature_columns = feature

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dict = super().__getitem__(idx)
        row = self.df.iloc[idx]

        if self.feature_columns:
            dict['features'] = torch.tensor(row[self.feature_columns].values.tolist(), dtype=torch.float32)
        return dict