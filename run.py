import sys
import ast
import torch
from Data import *
import pandas as pd
from helper import *
import torch.nn as nn
from BasicsModels import *
from TaxonomyModel import *
import torch.optim as optim
from transformers import AutoTokenizer, get_scheduler

# params change according to run
text_model_name = 'google-bert/bert-base-uncased'
code_model_name = 'microsoft/codebert-base'
path = "/home/nogaschw/Codeworkout/df_taxonomy_all.csv"
# feature = ['LogicBoolean','StringLen', 'StringEqual', 'CharEqual', 'ArrayIndex', 'DefFunction', 'IfKnowledge', 'StringKnowledge', 'LoopsKnowledge', 'Math+-*/%', 'LogicOperators']
feature = None
feature =['If/Else', 'NestedIf',
       'While', 'For', 'NestedFor', 'Math+-*/', 'Math%', 'LogicAndNotOr',
       'LogicCompareNum', 'LogicBoolean', 'StringFormat', 'StringConcat',
       'StringIndex', 'StringLen', 'StringEqual', 'CharEqual', 'ArrayIndex',
       'DefFunction']
#  feature = ['IfElse', 'Loops', 'MathOperations', 'LogicOperators', 'StringOperations', 'List', 'FileOperations', 'Functions', 'Dictionary', 'Tuple']

def get_df():
    global path
    df = pd.read_csv(path)
    df['prev_code'] = df['prev_code'].apply(ast.literal_eval)
    df['prev_question'] = df['prev_question'].apply(ast.literal_eval)
    df['score'] = df['score'].apply(ast.literal_eval)
    print(f'Create df...')
    return df

def run(df, dataset, model, name, lr, balance_def):
    global text_model_name
    global code_model_name
    global feature

    print(f"Start test {name}")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # Define tokenizers
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    code_tokenizer = AutoTokenizer.from_pretrained(code_model_name)
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    if code_tokenizer.pad_token is None:
        code_tokenizer.pad_token = code_tokenizer.eos_token

    # dataset
    train_dataloader, valid_dataloader, test_dataloader = create_data_loader(df, dataset, text_tokenizer, code_tokenizer, batch_size=1, balanced_def=balance_def, feature=feature, ids_filepath_prefix="/home/nogaschw/Codeworkout/Thesis/Data/codeworkout/split_ids")

    # model
    model = model.to(device)
    # pos_weight = torch.tensor([4.0]).to(device) # Adjust this based on your class imbalance
    criterion = nn.BCEWithLogitsLoss() #pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*10,) # avarage of epoch with early stop
    
    model = training_loop(model=model, train_dataloader=train_dataloader, test_dataloader=valid_dataloader, optimizer=optimizer, criterion=criterion, device=device, lr_scheduler=lr_scheduler, name=name)

    print(f"Checkpoint saved for model {name}")


if __name__ == '__main__':
    print(f'Starting...')

    hidden_size = 128
    num_layers = 2
    lr = 0.00001
    all_questions = int(sys.argv[1])

    taxonomy = int(sys.argv[2])

    balance_def = int(sys.argv[3])

    try: 
        mean = int(sys.argv[4])
    except:
        mean = 0

    df = get_df()
    balance_def = [oversampling_boosting, oversampling_df, undersampling_df, oversampling_augmentation][balance_def]
    
    name = f'1{["", "Mean_"][mean]}{"All_History_with"}_{["Current_Question", "All_Question"][all_questions]}_{hidden_size}_{num_layers}_{lr}_{balance_def.__name__}'
    
    ds_choice = [DatasetCodeQuestion, DatasetCodeQPrevQ][all_questions]

    model_choice = [CodeQuestionLSTMModel(text_model_name, code_model_name, hidden_size, num_layers),
                    CodeQPrevQLSTMModel(text_model_name, code_model_name, hidden_size, num_layers, mean=mean)][all_questions]
    
    if taxonomy:
        ds_choice = FeatureDataset
        model_choice = ModelComputationalConstructs(model_choice, hidden_size, len(feature))
        name = f'1Taxonomycodeworkout_{["Current_Question", "All_Question"][all_questions]}_{hidden_size}_{num_layers}_{lr}_{balance_def.__name__}'
    
    print(name, len(feature))
    run(df, ds_choice, model_choice, name, lr, balance_def)