import os
import torch
import pickle
import random
import pandas as pd
from sklearn.utils import resample
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


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

def add_snapshots(df, ratio=4):
    def _func(prev_score, prev_code, prev_label, num_attempts, len_code):
        if max([len(i) for i in prev_code]) > len_code:
            prev_code = [i[-len_code:] for i in prev_code]
            l = [True if not label and max(prev_score[i]) != 1 and len_code <= num_attempts[i] else label for i, label in enumerate(prev_label)]
            return prev_code, prev_label
        return None, None
    df_big = df.copy()
    for i in range(1, 10):
        df_new = df.copy()
        df_new['prev_code'], df_new['prev_label'] =  zip(*df_new.apply(lambda row: _func(row['prev_score'], row['prev_code'], row['prev_label'], row['num_attempts'], i), axis=1))
        df_new = df_new[df_new['prev_code'].notna()]
        df_new.reset_index(drop=True, inplace=True)
        df_big = pd.concat([df_big, df_new], ignore_index=True)
    return df_big

def same_df(df, ratio=4):
    return df

def oversampling_df(df, ratio=1):
    struggling_df = df[df['Label'] == 0]
    not_struggling_df = df[df['Label'] == 1]  
    max_len = len(not_struggling_df)
    struggling_upsampled = resample(struggling_df,random_state=42,n_samples=max_len//ratio,replace=True)
    not_struggling_upsampled = resample(not_struggling_df,random_state=42,n_samples=max_len,replace=True)
    print(len(struggling_upsampled),len(not_struggling_upsampled))
    return pd.concat([struggling_upsampled,not_struggling_upsampled])

def create_data_loader(df, dataset, text_tokenizer=None, code_tokenizer=None, batch_size=8, balanced_def=undersampling_df, ratio=1, limit=0, feature=None,
                       ids_filepath_prefix='/home/nogaschw/codeworkout/Thesis/split_ids', padding_size_code=2200, padding_size_q=200, create_split=False):        
    # Split the data to train and test by student ID
    if not create_split:
        print("Load exist spliting")
        train_ids, valid_ids, test_ids = load_ids(ids_filepath_prefix)
    else:
        student_id = df['student_id'].unique()
        id_to_struggle = df.groupby('student_id')['Label'].first()
        train_ids, test_ids = train_test_split(student_id, test_size=0.3, stratify=id_to_struggle[student_id])
        valid_ids, test_ids = train_test_split(test_ids, test_size=0.2/0.3, stratify=id_to_struggle[test_ids])

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