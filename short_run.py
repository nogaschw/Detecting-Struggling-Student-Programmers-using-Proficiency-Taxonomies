import warnings
warnings.filterwarnings('ignore') # Ignore all warnings
import wandb
import torch
from Data import *
from helper import *
from Models import *
import torch.nn as nn
from choosedataset import *
import Data_Embedding
from sklearn.metrics import roc_curve
from transformers import AutoTokenizer

i = 3
d = 1
lr = 0.0001
batch_size = 32
weights = None #[0.244, 0.756]
feature_dataset = Data_Embedding.FeatureDataset

data = [Codeworkout, Falcon, Both][d](also50unti80=False)
type = ['without taxonomy and past snapshots', 'without taxonomy', 'without snapshots', ''][i]
model = [AllWithLastSnapshot, All, AllWithLastSnapshotTax, AllWithTaxonomyQmatrix][i]
name = f"My full model opp label {type}"
padding_size_code = 100
feature = ['IfElse', 'Loops', 'MathOperations', 'LogicOperators', 'StringOperations', 'List', 'FileOperations', 'Functions', 'Dictionary', 'Tuple'] #, 'comprehension', 'alg_design', 'decomposition']
text_model_name = 'google-bert/bert-base-uncased'
code_model_name = 'microsoft/codebert-base'
df = data.df
Data_Embedding.load(data.embedded)
# change label
df['prev_label'] = df['prev_label'].apply(lambda x: [not i for i in x])
df['Label'] = df['Label'].apply(lambda x: not x)

# start
if type == 'without taxonomy and past snapshots' or type == 'without snapshots':
    df['prev_code'] = df['prev_code'].apply(lambda x: [[i[-1]] for i in x]) # Only submission
    padding_size_code = 1
else: 
    df['prev_code'] = df['prev_code'].apply(lambda x: [i[-padding_size_code:] for i in x]) # n submissions padding_size_code snapshots

wandb.login()

runner = wandb.init(
        project="Struggling Students",
        name=name,
        tags=[data.name, 'struggle_label'],
    )

print(name, runner.tags)
print(df['Label'].value_counts(), flush=True)
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
code_tokenizer = AutoTokenizer.from_pretrained(code_model_name)
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token
if code_tokenizer.pad_token is None:
    code_tokenizer.pad_token = code_tokenizer.eos_token

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

dataset = feature_dataset if type == '' or type == 'without snapshots' else Data_Embedding.Dataset_Embedding_Q

train_dataloader, valid_dataloader, test_dataloader = create_data_loader(df, dataset, text_tokenizer, code_tokenizer, balanced_def=same_df, limit=30, padding_size_q=30, padding_size_code=padding_size_code, batch_size=batch_size, create_split=False, ids_filepath_prefix='/home/nogaschw/Codeworkout/Thesis/Data/both/split_ids', feature=feature)
print(len(train_dataloader), len(valid_dataloader), len(test_dataloader), flush=True)
print(train_dataloader.dataset.df['Label'].value_counts())
print(valid_dataloader.dataset.df['Label'].value_counts())
print(test_dataloader.dataset.df['Label'].value_counts())
print(len(set(train_dataloader.dataset.df['student_id'])), len(set(valid_dataloader.dataset.df['student_id'])), len(set(test_dataloader.dataset.df['student_id'])))

model = model(text_model_name)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

model = model.to(device)
print(model)
checkpoint_dir = f'/home/nogaschw/Codeworkout/Thesis/Models/{data.name}'
name =  os.path.join(checkpoint_dir, f"{name}.pth")

model = training_loop(model=model, train_dataloader=train_dataloader, test_dataloader=valid_dataloader, optimizer=optimizer, criterion=criterion, device=device, name=name, weights=weights)

all_labels, all_probs = eval_loop(model, valid_dataloader, device)

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

J = tpr - fpr
best_index = J.argmax()
best_threshold = thresholds[best_index]

y_labels, y_probs = eval_loop(model, test_dataloader, device)
results(0.5, y_labels, y_probs)
results(best_threshold, y_labels, y_probs)

wandb.finish()