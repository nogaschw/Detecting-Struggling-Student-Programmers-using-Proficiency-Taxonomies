import os
import sys
import torch
import random
from helper import *
from Models import *
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
sys.path.append(os.path.join(os.getcwd(), 'Codeworkout/Thesis'))
from Data.Data import *
from Data.choosedataset import *
import Data.Data_Embedding as Data_Embedding
from sklearn.metrics import roc_curve
from transformers import AutoTokenizer

wandb.login()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set the configuration
config = Config()
d = ["codeworkout", "falcon", "Both"][config.dataset]

runner = wandb.init(
        project="Struggling Students",
        name=config.config_name,
        tags=[d, 'merged model']
    )

# Determine if latents should be used based on the model type
latents = True if config.run_model == 'PTM' and config.gemma_path is None else False

# Load the dataset
data = [Codeworkout, Falcon, Both][config.dataset](latents=latents, similarity=False, gemma_path=config.gemma_path)
df = data.df
print(df['skills_vec'].iloc[0])

# Load embeddings
Data_Embedding.load(data.embedded)

# Preprocess the data
df['prev_tasks'] = df['prev_tasks'].apply(lambda x: [i[-config.padding_size_code:] for i in x]) # n submissions padding_size_code snapshots

# Print label distribution
print(df['Label'].value_counts(), flush=True)

# Initialize the tokenizer
text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

# Set the device for computation
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

# Initialize the model, dataset, and loss functions based on the configuration
if config.run_model == 'PTM':
    dataset = Data_Embedding.DatasetPTMBoth
    model = PTM
    caculate_func = caculate_2losses
    criterion = [nn.BCEWithLogitsLoss(), nn.L1Loss()] 
else:
    dataset = Data_Embedding.DatasetOneLoss
    model = AblationStudy
    caculate_func = caculate_1loss
    criterion = nn.BCEWithLogitsLoss()

# Print label distribution again (redundant)
print(df['Label'].value_counts(), flush=True)

# Initialize the tokenizer again (redundant)
text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
if text_tokenizer.pad_token is None:
    text_tokenizer.pad_token = text_tokenizer.eos_token

# Set the device for computation again (redundant)
device_name = "cuda" if torch.cuda.is_available() else "cpu"

df['course_id'] = df['course_id'].apply(lambda x: 0 if np.isnan(x) else 1)

data_loaders_cw = create_data_loader_k_fold(df[df['course_id'] == 0], dataset, text_tokenizer,max_len_code=config.max_len_code, 
                                         padding_size_q=config.number_coding_tasks, padding_size_code=config.padding_size_code, seed=42,
                                         batch_size=config.batch_size)[0]

data_loaders_fc = create_data_loader_k_fold(df[df['course_id'] == 1], dataset, text_tokenizer,max_len_code=config.max_len_code, 
                                         padding_size_q=config.number_coding_tasks, padding_size_code=config.padding_size_code, seed=42,
                                         batch_size=config.batch_size)[0]

merged_dataset = ConcatDataset([data_loaders_fc[0].dataset, data_loaders_cw[0].dataset])
merged_loader = DataLoader(merged_dataset, batch_size=config.batch_size, shuffle=True)

data_loaders = [(merged_loader, (data_loaders_cw[1], data_loaders_fc[1]))]
# data_loaders = [(data_loaders_cw[0], data_loaders_fc[1]), (data_loaders_fc[0], data_loaders_cw[1]), (merged_loader, (data_loaders_cw[1], data_loaders_fc[1]))]

# Function to print data loader statistics
def num_of(train_dataloader, test_dataloader):
    print(len(train_dataloader), len(test_dataloader))

# Print statistics for each fold
for train, test in data_loaders:
    num_of(train, test)

# Initialize a dictionary to store fold results
fold_results = {'ROC-AUC' : []}

# Perform k-fold cross-validation
for fold, (train_dataloader, test_dataloader) in enumerate(data_loaders):
    print(f"Fold {fold + 1}:")    # Prepare data for current fold
    if config.run_model == 'PTM':
        m = model(config.text_model_name, config.max_len_code, len(np.unique(df['student_id_encoded']))) #, num_skills=16, num_prog_concepts=13)
    else:
        m = model(config.padding_size_code, config.text_model_name, config.max_len_code)
    print(m)
    optimizer = torch.optim.Adam(m.parameters(), lr=config.lr, weight_decay=1e-4)
   
    m = m.to(device)
    # Training Loop
    for epoch in range(config.epoch):
        total_loss = train_loop(m, train_dataloader, device, optimizer, criterion, caculate_func)

        if epoch % 10 == 0:
            print(f"Fold {fold + 1}, Epoch {epoch}: Loss = {total_loss / len(train_dataloader)}")
            wandb.log({'train_loss': (total_loss / len(train_dataloader))})

    if isinstance(test_dataloader, tuple):
        y_labels = []
        y_probs = []

        for i in test_dataloader:
            y_labels_single, y_probs_single = eval_loop(m, i, device, caculate_func=caculate_func)
            y_probs_single = np.array(y_probs_single)
            y_true_single = np.array(y_labels_single)
            print(roc_auc_score(y_true_single, y_probs_single))
            y_labels.extend(y_labels_single)
            y_probs.extend(y_probs_single)

    else:     # Evaluate the model on the test set for the current fold
        y_labels, y_probs = eval_loop(m, test_dataloader, device, caculate_func=caculate_func)
    y_prob = np.array(y_probs)
    y_true = np.array(y_labels)

        # Store the results for the current fold
    fold_results['ROC-AUC'].append(roc_auc_score(y_true, y_prob))

    # Print the results for the current fold
    for k, v in fold_results.items():
        wandb.log({f'fold{fold+1}_{k}': v[fold]})

# Aggregate and print the average results across all folds
for k, v in fold_results.items():
    wandb.log({f'{k}_avg': np.mean(v)})

wandb.finish()