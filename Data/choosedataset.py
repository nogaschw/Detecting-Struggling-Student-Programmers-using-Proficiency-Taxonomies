import re
import os
import ast
import sys
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.join(os.getcwd(), 'Codeworkout/Thesis'))
from Config import Config

# Read the programming concepts from a JSON file
with open('/home/nogaschw/Codeworkout/Thesis/programing_concepts.json', 'r') as file:
    programming_concepts= json.load(file)
config = Config()

def data_of_skills_according_gemma(df, path=None):
    print(path)
    if path == None:
        return df
    skills = pd.read_csv(path)[['student_id', 'decomposition', 'alg', 'reading']]
    skills['decomposition'] = skills['decomposition'].apply(lambda s: ast.literal_eval(s))
    skills['alg'] = skills['alg'].apply(lambda s: ast.literal_eval(s))
    skills['reading'] = skills['reading'].apply(lambda s: ast.literal_eval(s))
    skills['dec_avg'] = skills['decomposition'].apply(lambda x: np.mean([y/5 for y in x if y is not None]))
    skills['alg_avg'] = skills['alg'].apply(lambda x: np.mean([y/5 for y in x if y is not None]))
    skills['read_avg'] = skills['reading'].apply(lambda x: np.mean([y/5 for y in x if y is not None]))
    df = df.merge(skills, how='left', on='student_id')
    df['skills_vec'] = df.apply(lambda row: row['skills_vec'] + [row['dec_avg'], row['alg_avg'], row['read_avg']], axis=1)
    return df

def add_skills_vec(df, size=10, latents=False, similarity=False, path=None):
    """
    Create TBPP ground  to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'prev_comp_cons' and 'prev_score' columns.
        size (int): Size of the skills vector.
        latents (bool): Whether to include latent features.
    
    Returns:
        pd.DataFrame: DataFrame with added 'skills_vec' column.
    """
    def expand_lists(row):
        expanded = []
        for lst in row:  # Iterate over 30 lists
            expanded.append(lst + [1, 1, 1])  # Add three ones at the end
        return expanded
    # Apply the function to the column
    if latents:
        df['prev_comp_cons'] = df['prev_comp_cons'].apply(expand_lists)
    size = size + 3 if latents else size
    print(size, latents, "similarity:", similarity)
    df['score_vec'] = df['prev_scores'].apply(lambda x: [x[i][-1] if i < len(x) else 0 for i in range(30)])
    # df['score_vec'] = df['prev_scores'].apply(lambda x: [x[i][-1] * (0.95 ** len(x[i])) if i < len(x) else 0 for i in range(30)])
    print(df['score_vec'].iloc[0])
    df['score_vec'] = df.apply(lambda x: x['similarities'] * x['score_vec'], axis=1) if similarity else df['score_vec']
    df['skills_vec'] = df.apply(lambda row: np.dot(
        np.array(row.prev_comp_cons).T,
        np.array(row.score_vec).reshape(30, 1)
        ).reshape(size), 
        axis=1)
    # Normalize skills_vec
    all_skills_vec = np.vstack(df['skills_vec'])  # Combine all vectors into a 2D array
    min_vec = all_skills_vec.min(axis=0)  # Compute min for each skill across all rows
    max_vec = all_skills_vec.max(axis=0)  # Compute max for each skill across all rows
    range_vec = max_vec - min_vec  # Compute the range
    range_vec[range_vec == 0] = 1
    df['skills_vec'] = df['skills_vec'].apply(lambda vec: (vec - min_vec) / range_vec)
    df['skills_vec'] = df['skills_vec'].apply(lambda vec: [v if v > 0 else vec.mean() for v in vec])
    df = data_of_skills_according_gemma(df, path)
    return df 

def represent_question(questions, word):
    text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    text_model = AutoModel.from_pretrained(config.text_model_name)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    text_model = text_model.to(device)
    output_text = text_tokenizer(list(questions[word]), max_length=text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        text_output = text_model(output_text['input_ids'].to(device), output_text['attention_mask'].to(device)).last_hidden_state[:, 0, :]
    text_output = text_output.to('cpu')
    return text_output

def question_similarity(questions_dict, new_id, prev_ids, course_id=None):
    new_id = new_id if course_id is None else (course_id,new_id)
    target = questions_dict[new_id].reshape(1, -1)
    prev = np.stack([questions_dict[
        int(i) if course_id is None else (course_id,i)] 
                     for i in prev_ids])
    return cosine_similarity(target, prev)[0]

def student_id(df):
    """
    Encode student IDs as integers.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'student_id' column.
    
    Returns:
        pd.DataFrame: DataFrame with added 'student_id_encoded' column.
    """
    student_id_mapping = {sid: idx for idx, sid in enumerate(np.unique(df.student_id.values))}
    df["student_id_encoded"] = df["student_id"].map(student_id_mapping)
    return df
    
class Falcon:
    """
    Class to process Falcon dataset.
    
    Attributes:
        name (str): Name of the dataset.
        embedded (list): The paths to the codes after made process to with LLM
        df (pd.DataFrame): Processed DataFrame.
    """
    def __init__(self, latents=False, change_features=True, similarity=False, gemma_path=None):
        """
        Initialize the Falcon class.
        
        Args:
            latents (bool): Whether to include latent features.
            change_features (bool): Whether to change features based on programming concepts.
        """
        self.name = 'falcon'
        self.embedded = ['falcon_to_model_output']
        path = config.path_saved_falcon
        self.df = pd.read_pickle(path)
        self.df.fillna(0, inplace=True)    
        num_col, questions = self._prev_constracts(change_features)
        if similarity:
            questions_dict = {}
            q_rep = represent_question(questions, 'prompt')
            for i, (idx, q) in enumerate(questions[['course_id', 'id']].iterrows()):
                questions_dict[(q['course_id'], q['id'])] = q_rep[i]
            self.df['similarities'] = self.df.apply(lambda x: question_similarity(questions_dict, x['new_task_id'], x['prev_tasks_id'], x['course_id']), axis=1)
        self.df = student_id(self.df)
        self.df = add_skills_vec(self.df, num_col, latents, similarity, gemma_path)
    
    def _prev_constracts(self, change_features):
        """
        Process previous constructs and update the DataFrame.
        """
        cleaned_file_path = config.falconcode_questions_path
        questions = pd.read_csv(cleaned_file_path, sep=',')
        questions = questions[questions.columns[1:]]
        questions['prompt'] = questions['prompt'].apply(lambda x: x.split('PROBLEM STATEMENT:')[-1] if x.__contains__("PROBLEM STATEMENT:") else x)
        questions['prompt'] = questions.prompt.apply(lambda text: re.sub(r'\bPROBLEM STATEMENT: \b', '', text).strip())
        questions['id'] = questions['id'].apply(lambda x: x.lower())
        questions = questions[(questions['type'] == 'skill') | (questions['type'] == 'lab')]

        q_with_compu = questions.copy()
        if change_features:
            rem_col = []
            for k in programming_concepts['falcon']:
                q_with_compu[k] = q_with_compu[programming_concepts['falcon'][k]].max(axis=1)
                rem_col.extend(programming_concepts['falcon'][k])
            q_with_compu = q_with_compu.drop(columns=rem_col)
        q_with_compu.fillna(0, inplace=True)
        cols_to_merge = q_with_compu.columns[7:]
        q_with_compu['comp_cons'] = q_with_compu[cols_to_merge].apply(lambda row: row.values.tolist(), axis=1)
        questions_df = q_with_compu.drop(columns=cols_to_merge)
        questions_df['id'] = questions_df['id'].apply(lambda x: x.lower())
        Comp_cons_dict = questions_df.set_index('id')['comp_cons'].to_dict()
        self.df['prev_comp_cons'] = self.df['prev_tasks_id'].apply(lambda x: [Comp_cons_dict[i] for i in x])
        self.df['curr_comp_cons'] = self.df['new_task_id'].apply(lambda x: Comp_cons_dict[x])
        self.df.rename(columns={'prompt': 'question'}, inplace=True)
        return len(cols_to_merge), questions[['id', 'course_id', 'prompt']]
    
class Codeworkout:
    """
    Class to process Codeworkout dataset.
    
    Attributes:
        name (str): Name of the dataset.
        embedded (list):  The paths to the codes after made process to with LLM.
        df (pd.DataFrame): Processed DataFrame.
    """
    def __init__(self, latents=False, change_features=True, similarity=False, gemma_path=None):
        """
        Initialize the Codeworkout class.
        
        Args:
            latents (bool): Whether to include latent features.
            change_features (bool): Whether to change features based on programming concepts.
        """
        self.name = 'codeworkout'
        self.embedded = ['codeworkout_to_model_output']
        df = pd.read_pickle(config.path_saved_codeworkout)
        df.fillna(0, inplace=True)
        questions_df = pd.read_excel(config.codeworkout_questions_path)
        questions_df.fillna(0, inplace=True)
        questions_df = self._order_features(questions_df) if change_features else questions_df
        cols_to_merge = questions_df.columns.tolist()[3:]
        questions_df['comp_cons'] = questions_df[cols_to_merge].apply(lambda row: row.values.tolist(), axis=1)
        questions_df = questions_df.drop(columns=cols_to_merge)
        Comp_cons_dict = questions_df.set_index('ProblemID')['comp_cons'].to_dict()
        df['prev_comp_cons'] = df['prev_tasks_id'].apply(lambda x: [Comp_cons_dict[int(x[i])] if i < len(x) else [0 for i in range(10)] for i in range(30)])
        print(df['prev_comp_cons'].apply(len).value_counts())
        df['curr_comp_cons'] = df['new_task_id'].apply(lambda x: Comp_cons_dict[x])
        if similarity:
            questions_dict = {}
            q_rep = represent_question(questions_df, 'Requirement')
            for i, id in enumerate(questions_df['ProblemID']):
                questions_dict[id] = q_rep[i]
            df['similarities'] = df.apply(lambda x: question_similarity(questions_dict, x['new_task_id'], x['prev_tasks_id']), axis=1)
        self.df = student_id(df)
        self.df = add_skills_vec(df, len(cols_to_merge), latents, similarity, gemma_path)

    def _order_features(self, df):
        """
        Order features based on programming concepts.
        """
        rem_col = []
        for k in programming_concepts['codeworkout']:
            df[k] = df[programming_concepts['codeworkout'][k]].max(axis=1)
            rem_col.extend(programming_concepts['codeworkout'][k])
        df = df.drop(columns=rem_col)
        df.fillna(0, inplace=True)
        return df
    
class Both:
    def __init__(self, latents=False, change_features=True, similarity=False, gemma_path=None):
        self.name = "both"
        self.cw = Codeworkout(latents, change_features, similarity, gemma_path[0])
        self.fc = Falcon(latents, change_features, similarity, gemma_path[1])
        self.embedded = self.cw.embedded
        self.embedded.extend(self.fc.embedded)
        self.df = pd.concat([self.cw.df, self.fc.df], ignore_index=True)
