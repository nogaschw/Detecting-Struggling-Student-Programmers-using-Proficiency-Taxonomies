import os
import sys
import wandb
import torch
import random
from helper import *
from Models import *
import torch.nn as nn
sys.path.append(os.path.join(os.getcwd(), 'Codeworkout/Thesis/Data'))
from Data import *
from sklearn.metrics import roc_curve
from transformers import AutoTokenizer
from choosedataset import *
import Data_Embedding as Data_Embedding

wandb.login()

# Set the configuration
config = Config()
comp_name = config.config_name

runner = wandb.init(
        project="Struggling Students",
        name=comp_name,
        tags=["gemma"],
    )

# Determine if latents should be used based on the model type
latents = True if config.run_model == 'PTM' and config.gemma_path is None else False

# Load the dataset
data = [Codeworkout, Falcon, Both][config.dataset](latents=latents, similarity=False, gemma_path=config.gemma_path)
df = data.df

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
print(config.run_model)
# Initialize the model, dataset, and loss functions based on the configuration
if config.run_model == 'PTM':
    dataset = Data_Embedding.DatasetPTM
    model = PTM(config.text_model_name, config.max_len_code, len(np.unique(df['student_id_encoded'])))
    caculate_func = caculate_2losses
    criterion = [nn.BCEWithLogitsLoss(), nn.L1Loss()] 
else:
    dataset = Data_Embedding.DatasetOneLoss
    model = AblationStudy(config.padding_size_code, config.text_model_name, config.max_len_code)
    caculate_func = caculate_1loss
    criterion = nn.BCEWithLogitsLoss()

# Create data loaders for training, validation, and testing
train_dataloader, valid_dataloader, test_dataloader = create_data_loader(df, dataset, text_tokenizer, max_len_code=config.max_len_code, padding_size_q=config.number_coding_tasks, 
                                                                         padding_size_code=config.padding_size_code, batch_size=config.batch_size, create_split=False,
                                                                         ids_filepath_prefix=f'/home/nogaschw/Codeworkout/UseData/both/')

# Print the sizes of the data loaders and label distributions
print(len(train_dataloader), len(valid_dataloader), len(test_dataloader), flush=True)
print(train_dataloader.dataset.df['Label'].value_counts())
print(valid_dataloader.dataset.df['Label'].value_counts())
print(test_dataloader.dataset.df['Label'].value_counts())
print(len(set(train_dataloader.dataset.df['student_id'])), len(set(valid_dataloader.dataset.df['student_id'])), len(set(test_dataloader.dataset.df['student_id'])))

# Print the intersections of student IDs between the splits
print(set(train_dataloader.dataset.df['student_id']).intersection(set(valid_dataloader.dataset.df['student_id'])))
print(set(train_dataloader.dataset.df['student_id']).intersection(set(test_dataloader.dataset.df['student_id'])))
print(set(valid_dataloader.dataset.df['student_id']).intersection(set(test_dataloader.dataset.df['student_id'])))

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)

save_model_path = f'/home/nogaschw/Codeworkout/SavedModel/{config.config_name}.pth'
# Move the model to the device
model = model.to(device)

print(model)
# Train the model
print("Running UPDATED run_train_val_test.py")

model = training_loop(model=model, train_dataloader=train_dataloader, test_dataloader=valid_dataloader, optimizer=optimizer, 
                      criterion=criterion, device=device, name=save_model_path, caculate_func=caculate_func)

# Evaluate the model on the validation set
all_labels, all_probs = eval_loop(model, valid_dataloader, device, caculate_func=caculate_func)

# Calculate the ROC curve and find the best threshold
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
J = tpr - fpr
best_index = J.argmax()
best_threshold = thresholds[best_index]

# Evaluate the model on the test set
y_labels, y_probs = eval_loop(model, test_dataloader, device, caculate_func=caculate_func)

# Print the results
results(0.5, y_labels, y_probs)
results(best_threshold, y_labels, y_probs)

y_pred = np.where(y_probs > 0.5, 1, 0)
df_results = pd.DataFrame({
    'y_true': y_labels.squeeze(),
    'y_probs': y_probs.squeeze(),
    'y_pred': y_pred.squeeze()
})

df_results.to_csv(f'/home/nogaschw/Codeworkout/Thesis/{comp_name}.csv')

wandb.finish()