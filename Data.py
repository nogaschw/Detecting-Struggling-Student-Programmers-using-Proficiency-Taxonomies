import os
import torch
import pickle
import random
import pandas as pd
from sklearn.utils import resample
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class DatasetMean(Dataset):
    def __init__(self, df, text_tokenizer, code_tokenizer, max_len_text=512, max_len_code=512, limit_len=0, feature=None):
        self.df = df
        self.text_tokenizer = text_tokenizer
        self.code_tokenizer = code_tokenizer
        self.max_len_text = max_len_text
        self.max_len_code = max_len_code

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['question']
        code_samples = row['prev_code']
        label = torch.tensor(row['struggling'], dtype=torch.float)

        flat_code_sampels = []
        
        for i in code_samples:
            flat_code_sampels.extend(i)

        # Tokenize question text
        text_inputs = self.text_tokenizer(question, max_length=self.max_len_text, padding='max_length', truncation=True, return_tensors='pt')

        # Tokenize code samples
        code_inputs = [self.code_tokenizer(code_sample, max_length=self.max_len_code, padding='max_length', truncation=True, return_tensors='pt') for code_sample in flat_code_sampels]
        # Stack and compute mean
        input_ids_stack = torch.stack([data['input_ids'] for data in code_inputs]).squeeze(1)
        code_input_ids = torch.mean(input_ids_stack.float(), dim=0, keepdim=True)
        attention_mask_stack = torch.stack([data['attention_mask'] for data in code_inputs])
        code_attention_mask = torch.max(attention_mask_stack.float(), dim=0).values

        return {
            'text_input_ids': text_inputs['input_ids'].squeeze(0),
            'text_attention_mask': text_inputs['attention_mask'].squeeze(0),
            'code_input_ids': code_input_ids.squeeze(0).long(),
            'code_attention_mask': code_attention_mask.squeeze(0).long(),
            'label': label
        }
    
class DatasetLimitMean(Dataset):
    def __init__(self, df, text_tokenizer, code_tokenizer, num_question=5, max_len_code=512, limit_len=0, padding_size_code=1000, padding_size_q=50, feature=None):
        self.df = df
        self.df = df[df['prev_code'].apply(len) >= 5]
        self.code_tokenizer = code_tokenizer
        self.max_len_code = max_len_code
        self.num_question = num_question

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code_samples = row['prev_code']
        label = torch.tensor(row['struggling'], dtype=torch.float)

        coding_task_code_input_ids = []
        coding_task_code_attention_mask = []

        for task_code in code_samples[-self.num_question:]:
            snapshots = []
            for snapshot in task_code:
                snapshots.append(snapshot)

            # Tokenize code of task
            code_inputs = [self.code_tokenizer(code_sample, max_length=self.max_len_code, padding='max_length', truncation=True, return_tensors='pt') for code_sample in snapshots]
            
            # Stack and compute mean
            input_ids_stack_mean = torch.stack([data['input_ids'] for data in code_inputs]).squeeze(1)
            input_ids_stack_mean = torch.mean(input_ids_stack_mean.float(), dim=0, keepdim=True)

            attention_mask_mean = torch.stack([data['attention_mask'] for data in code_inputs])
            attention_mask_mean = torch.max(attention_mask_mean.float(), dim=0).values
            coding_task_code_input_ids.append(input_ids_stack_mean)
            coding_task_code_attention_mask.append(attention_mask_mean)
        
        return {
            'code_input_ids': torch.stack(coding_task_code_input_ids).squeeze(1).long(),
            'code_attention_mask': torch.stack(coding_task_code_attention_mask).squeeze(1).long(),
            'label': label
        }

def tokenize_vec(tokenizer, samples, max_len, limit_len, padding_size):
    if limit_len: # diffrent from 0
        if len(samples) > limit_len:
            samples = samples[-limit_len:]
    if type((samples)[0]) is list:
        flat_code_sampels = []
        for i in samples:
            flat_code_sampels.extend(i)
        samples = flat_code_sampels

    inputs = [tokenizer(sample, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt') for sample in samples]
    input_ids_stack = torch.stack([data['input_ids'] for data in inputs]).squeeze(1)
    attention_mask_stack = torch.stack([data['attention_mask'] for data in inputs]).squeeze(1)
    
    code_num = input_ids_stack.size(0)
    padded_input_ids = torch.zeros((padding_size, max_len), dtype=torch.long)
    padded_attention_mask = torch.zeros((padding_size, max_len), dtype=torch.long)
    if code_num <= padding_size:  # No truncation needed, only padding
        padded_input_ids[:code_num, :] = input_ids_stack
        padded_attention_mask[:code_num, :] = attention_mask_stack
    else:  # Truncate to fit
        code_num = padding_size
        padded_input_ids = input_ids_stack[-padding_size:, :]
        padded_attention_mask = attention_mask_stack[-padding_size:, :]
    return padded_input_ids, padded_attention_mask, code_num

    
class DatasetCodeQuestion(Dataset):
    def __init__(self, df, text_tokenizer, code_tokenizer, max_len_text=512, max_len_code=512, limit_len=0, padding_size_code=1000, padding_size_q=50, feature=None):
        self.df = df
        self.text_tokenizer = text_tokenizer
        self.code_tokenizer = code_tokenizer
        self.max_len_text = max_len_text
        self.max_len_code = max_len_code
        self.limit_len = limit_len
        self.padding_size_code = padding_size_code

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['question']
        code_samples = row['prev_code']
        label = torch.tensor(row['struggling'], dtype=torch.float)

        # Tokenize question text
        text_inputs = self.text_tokenizer(question, max_length=self.max_len_text, padding='max_length', truncation=True, return_tensors='pt')

        # Tokenize code samples
        input_ids_stack, attention_mask_stack, code_num = tokenize_vec(self.code_tokenizer, code_samples, self.max_len_code, self.limit_len, self.padding_size_code)
            
        return {
            'text_input_ids': text_inputs['input_ids'].squeeze(0),
            'text_attention_mask': text_inputs['attention_mask'].squeeze(0),
            'code_input_ids': input_ids_stack,
            'code_attention_mask': attention_mask_stack,
            'code_num': code_num, 
            'label': label
        }
    
class DatasetCodeQPrevQ(DatasetCodeQuestion):
    def __init__(self, df, text_tokenizer, code_tokenizer, max_len_text=512, max_len_code=512, limit_len=0, padding_size_code=1000, padding_size_q=50,feature=None):
        super().__init__(df, text_tokenizer, code_tokenizer, max_len_text=512, max_len_code=512, limit_len=0, padding_size_code=1000, padding_size_q=50, feature=None)
        self.padding_size_q = padding_size_q
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dict = super().__getitem__(idx)
        row = self.df.iloc[idx]
        prev_question = row['prev_question']
        code_samples = row['prev_code'][::-1]
        len_snapshots = []

        for i, c in enumerate(code_samples):
            len_snapshots.append(len(c))
            if i == self.limit_len - 1 or i == self.padding_size_q - 1:
                break

        for i in range(self.padding_size_q - len(len_snapshots)):
            len_snapshots.append(0)
        
        dict['code_input_ids'], dict['code_attention_mask'], dict['code_num'] = tokenize_vec(self.code_tokenizer, code_samples, self.max_len_code, self.padding_size_q, self.padding_size_code)
        question_input_ids_stack, question_attention_mask_stack, que_num = tokenize_vec(self.text_tokenizer, prev_question, self.max_len_text, self.limit_len, self.padding_size_q)
        
        dict['questions_input_ids'] = question_input_ids_stack
        dict['questions_attention_mask']= question_attention_mask_stack
        dict['num_snapshots'] = torch.tensor(len_snapshots)
        dict['que_num'] = que_num
        return dict

class FeatureDataset(DatasetCodeQPrevQ):
    def __init__(self, df, text_tokenizer, code_tokenizer, max_len_text=512, max_len_code=512, limit_len=0, padding_size_code=1000, padding_size_q=50 ,feature=None):
        super().__init__(df, text_tokenizer, code_tokenizer, max_len_text=512, max_len_code=512, padding_size_code=1000, padding_size_q=50, limit_len=0)
        self.feature_columns = feature

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dict = super().__getitem__(idx)
        row = self.df.iloc[idx]

        if self.feature_columns:
            dict['features'] = torch.tensor(row[self.feature_columns].values.tolist(), dtype=torch.float32)
            
        return dict
    

def oversampling_boosting(df, ratio=None):
    def _sample_from_code_array(code_array):
        n = len(code_array)    
        selected_snippets = [True if random.random() < ((3 * i) // n + 1) * 0.2 else False for i, codes in enumerate(code_array)]
        if not any(selected_snippets):
            return _sample_from_code_array(code_array)
        return selected_snippets
    new_rows = []

    def _filter_by_boolean_mask(boolean_array, object_array):
        return [obj for keep, obj in zip(boolean_array, object_array) if keep]

    for index, row in df.iterrows():
        num_additional_rows = 4 if row['struggling'] == 1 else 0
        if len(row['prev_code']) >= 3:
            for _ in range(num_additional_rows):
                new_row = row.copy()
                bool_arr = _sample_from_code_array(row['prev_code'])
                new_row['prev_code'] = _filter_by_boolean_mask(bool_arr, new_row['prev_code'])
                new_row['prev_question'] = _filter_by_boolean_mask(bool_arr, new_row['prev_question'])
                # new_row['score'] = _filter_by_boolean_mask(bool_arr, new_row['score'])
                new_rows.append(new_row)
        
    additional_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([df, additional_df], ignore_index=True)
    print(combined_df["struggling"].value_counts(), flush=True)
    return combined_df

def oversampling_augmentation(df, ratio=None):
    def _create_new_row(row, cut_inx):
        new_row = row.copy()
        new_row['prev_code'] = new_row['prev_code'][-cut_inx:]
        new_row['prev_question'] = new_row['prev_question'][-cut_inx:]
        # new_row['score'] = new_row['score'][-cut_inx:]
        return new_row

    new_rows = []
    for index, row in df.iterrows():
        num_question = len(row['prev_code'])
        num_additional_rows = 4 if row['struggling'] == 1 else 0
        if len(row['prev_code']) < num_additional_rows:
            for i in range(1, num_question):
                new_rows.append(_create_new_row(row, i))            
        else:
            if num_additional_rows != 0:
                for i in range(num_question // (num_additional_rows + 1) + 1 , num_question, num_question // (num_additional_rows + 1) + 1):
                    new_rows.append(_create_new_row(row, i))
                
    additional_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([df, additional_df], ignore_index=True)
    print(combined_df["struggling"].value_counts(), flush=True)
    return combined_df

def undersampling_df(df, ratio=1):
    struggling_df = df[df['struggling'] == 1]
    not_struggling_df = df[df['struggling'] == 0]
    min_len = len(struggling_df)
    return pd.concat([not_struggling_df.sample(min_len * ratio), struggling_df.sample(min_len)]).sample(frac=1)

def same_df(df, ratio=4):
    return df

def oversampling_df(df, ratio=1):
    struggling_df = df[df['struggling'] == 1]
    not_struggling_df = df[df['struggling'] == 0]  
    max_len = len(not_struggling_df)
    struggling_upsampled = resample(struggling_df,random_state=42,n_samples=max_len//ratio,replace=True)
    not_struggling_upsampled = resample(not_struggling_df,random_state=42,n_samples=max_len,replace=True)
    return pd.concat([struggling_upsampled,not_struggling_upsampled])

def create_data_loader(df, dataset, text_tokenizer=None, code_tokenizer=None, batch_size=8, balanced_def=undersampling_df, ratio=1, limit=0, feature=None,
                       ids_filepath_prefix='/home/nogaschw/codeworkout/Thesis/split_ids', padding_size_code=2200, padding_size_q=200, create_split=False):        
    # Split the data to train and test by student ID
    if not create_split:
        print("Load exist spliting")
        train_ids, valid_ids, test_ids = load_ids(ids_filepath_prefix)
    else:
        student_id = df['student_id'].unique()
        id_to_struggle = df.groupby('student_id')['struggling'].first()
        train_ids, test_ids = train_test_split(student_id, test_size=0.3, random_state=42, stratify=id_to_struggle[student_id])
        valid_ids, test_ids = train_test_split(test_ids, test_size=0.2/0.3, random_state=42, stratify=id_to_struggle[test_ids])

    train_df = df[df['student_id'].isin(train_ids)]
    valid_df = df[df['student_id'].isin(valid_ids)]
    test_df = df[df['student_id'].isin(test_ids)]

    # Balance training set
    train_df_balance = balanced_def(train_df, ratio)
    
    # Tokenize
    train_dataset = dataset(train_df_balance, text_tokenizer, code_tokenizer, feature=feature, padding_size_code=padding_size_code, padding_size_q=padding_size_q)
    valid_dataset = dataset(valid_df, text_tokenizer, code_tokenizer, feature=feature, padding_size_code=padding_size_code, padding_size_q=padding_size_q)
    test_dataset = dataset(test_df, text_tokenizer, code_tokenizer, limit_len=limit, feature=feature, padding_size_code=padding_size_code, padding_size_q=padding_size_q)

    # Dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)       
    return train_dataloader, valid_dataloader, test_dataloader

def save_ids(train_ids, valid_ids, test_ids, filepath_prefix='Data/split_ids'):
    with open(f'{filepath_prefix}_train_ids.pkl', 'wb') as f:
        pickle.dump(train_ids, f)
    with open(f'{filepath_prefix}_valid_ids.pkl', 'wb') as f:
        pickle.dump(valid_ids, f)
    with open(f'{filepath_prefix}_test_ids.pkl', 'wb') as f:
        pickle.dump(test_ids, f)

def load_ids(filepath_prefix='Data/split_ids'):
    with open(f'{filepath_prefix}_train_ids.pkl', 'rb') as f:
        train_ids = pickle.load(f)
    with open(f'{filepath_prefix}_valid_ids.pkl', 'rb') as f:
        valid_ids = pickle.load(f)
    with open(f'{filepath_prefix}_test_ids.pkl', 'rb') as f:
        test_ids = pickle.load(f)
    return train_ids, valid_ids, test_ids

