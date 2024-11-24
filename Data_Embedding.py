import torch
import pickle
from torch.utils.data import Dataset

base_path = '/home/nogaschw/Codeworkout/Thesis/Data/'
code_to_model_dict = None

def load(all):
    global code_to_model_dict
    print(all)
    for i in all:
        dict_list = []
        with open(base_path + i + '.pkl', 'rb') as file:
            dict_list.append(pickle.load(file))
    code_to_model_dict = {k: v for d in dict_list for k, v in d.items()}

class Dataset_Embedding_Q(Dataset):
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
        prev_questions = row['prev_question']
        label = torch.tensor(row['Label'], dtype=torch.float)
        q_num = len(prev_questions)
        
        # Tokenize curr coding text
        text_inputs = self.text_tokenizer(question, max_length=self.max_len_text, padding='max_length', truncation=True, return_tensors='pt')
        
        len_snapshots = [0 for i in range(q_num)]
        prev_questions_input = torch.zeros((q_num, self.max_len_text), dtype=torch.float)
        prev_questions_mask = torch.zeros((q_num, self.max_len_text), dtype=torch.float)
        embedding = torch.zeros((self.padding_size_q, self.padding_size_code, self.max_len_code), dtype=torch.float)

        for q_idx, codes in enumerate(code_samples):
            len_snapshots[q_idx] = len(codes)
            for c_idx, code in enumerate(codes):
                embedding[q_idx, c_idx , :] = torch.tensor(code_to_model_dict[code])
        
        for i, q in enumerate(prev_questions):
            q_inputs = self.text_tokenizer(q, max_length=self.max_len_text, padding='max_length', truncation=True, return_tensors='pt')
            prev_questions_input[i, :] = q_inputs['input_ids'].squeeze(0)
            prev_questions_mask[i, :] =  q_inputs['attention_mask'].squeeze(0)

        return {
            'text_input_ids': text_inputs['input_ids'].squeeze(0),
            'text_attention_mask': text_inputs['attention_mask'].squeeze(0),
            'code_embedding': embedding,
            'prev_q_input_ids': prev_questions_input.long(),
            'prev_q_attention_mask': prev_questions_mask.long(),
            'code_num': torch.tensor(len_snapshots),
            'prev_struggling': torch.tensor(row['prev_label']),
            'label': label
        }
    
class FeatureDataset(Dataset_Embedding_Q):
    def __init__(self, df, text_tokenizer, code_tokenizer, max_len_text=512, max_len_code=768, limit_len=0, padding_size_code=2000, padding_size_q=200, feature=None):
        super().__init__(df, text_tokenizer, code_tokenizer, max_len_text, max_len_code, limit_len, padding_size_code, padding_size_q)
        self.feature_columns = feature

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dict = super().__getitem__(idx)
        row = self.df.iloc[idx]

        if self.feature_columns:
            dict['features'] = torch.tensor(row[self.feature_columns].values.tolist(), dtype=torch.float32)
            
        return dict
    
class LastFeatureDataset(Dataset_Embedding_Q):
    def __init__(self, df, text_tokenizer, code_tokenizer, max_len_text=512, max_len_code=768, limit_len=0, padding_size_code=2000, padding_size_q=200, feature=None):
        super().__init__(df, text_tokenizer, code_tokenizer, max_len_text, max_len_code, limit_len, padding_size_code, padding_size_q)

    def __getitem__(self, idx):
        dict = super().__getitem__(idx)
        row = self.df.iloc[idx]
        dict['prev_features'] = torch.tensor(row['prev_comp_cons'], dtype=torch.float32)     
        return dict