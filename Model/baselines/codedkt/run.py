import os
import sys
from CodeDKT import *
from torch.utils.data import DataLoader
from readdata import data_reader, StudentDataset
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../Data")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from Data import *
from choosedataset import *
from helper import * 
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

class Config:
    def __init__(self):
        self.length = 100
        self.lr = 0.0001
        self.bs = 32
        self.epochs = 15
        self.hidden = 128
        self.layers = 1
        self.code_path_length = 8
        self.code_path_width = 2
        self.dataset = 1
        self.padding_size_code = 100

def create_questions_dict(df):
    all_future_q = set()
    for i in df['new_task_id']:
        all_future_q.add(i)

    all_prev_q = set()
    for i in df['prev_tasks_id']:
        all_prev_q = all_prev_q.union(set(i))
    all_problems = all_future_q.union(all_prev_q)
    return {name: idx for idx, name in enumerate(all_problems)}

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

config = Config()

data = [Codeworkout, Falcon][config.dataset]()
df = data.df
code_df = pd.read_csv("/home/nogaschw/Codeworkout/UseData/labeled_paths_python_all.tsv",sep="\t")
df['prev_tasks'] = df['prev_tasks'].apply(lambda x: [i[-config.padding_size_code:] for i in x]) # n submissions padding_size_code snapshots

code_df.rename(columns={'student_id': 'SubjectID'}, inplace=True)
code_df.rename(columns={'clean_code': 'Code'}, inplace=True)
question_dict = create_questions_dict(df)

def caculate_1loss(batch, model, device, criterion, loss_fn=None):
    dict_batch = {k: v.to(device) for k, v in batch.items()}
    model_params = {k: v for k, v in dict_batch.items() if k != 'label'}
    logits = model(*model_params.values())
    label = dict_batch['label'].float()
    if not criterion:
        return logits[1], label
    loss = criterion(logits, batch['row'], label)
    del dict_batch, model_params, logits, label
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return loss

caculate_func = caculate_1loss
criterion = lossFunc(len(question_dict), config.length, device)

def create_data_loader_k_fold(df, dataset, question_dict=None, batch_size=32, k=5):            # Setup k-fold
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    student_id = df['student_id'].unique()
    id_to_struggle = df.groupby('student_id')['Label'].first()
    data_loaders = []

    # Perform k-fold split
    for train_idx, test_idx in kf.split(student_id, id_to_struggle[student_id]):
        train_students = student_id[train_idx]
        test_students = student_id[test_idx]
        handler = data_reader(df, code_df, question_dict, config.length, config.questions)
        handler.get_data(train_students, test_students)
        # Create train and test DataFrames
        train_df = df[df['student_id'].isin(train_students)]
        test_df = df[df['student_id'].isin(test_students)]

        # Tokenize
        train_dataset = dataset(train_df, handler)
        test_dataset = dataset(test_df, handler, "test")

        # Create DataLoaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Append to results
        data_loaders.append((train_dataloader, test_dataloader))
    return data_loaders

config.questions = len(question_dict)
data_loaders = create_data_loader_k_fold(df, StudentDataset, question_dict, batch_size=config.bs)

# Initialize a dictionary to store fold results
fold_results = {'ROC-AUC' : [], 'f1' : [], 'recall': [], "precision": [], 'calibration': [], 'f1-0.5': [], 'recall-0.5': [], 'precision-0.5': []}

# Perform k-fold cross-validation
for fold, (train_dataloader, test_dataloader) in enumerate(data_loaders):
    print(f"Fold {fold + 1}:")    # Prepare data for current fold
    node_count = train_dataloader.dataset.node_count
    path_count = train_dataloader.dataset.path_count

    m = c2vRNNModel(config.questions * 2,
                    config.hidden,
                    config.layers,
                    config.questions,
                    node_count, path_count, device) 
    loss_fn = None
    optimizer = torch.optim.Adam(m.parameters(), lr=config.lr, weight_decay=1e-4)
   
    m = m.to(device)
    # Training Loop
    for epoch in range(15):
        helperconfig.total_binary_loss = 0
        helperconfig.total_skills_loss = 0
        total_loss = train_loop(m, train_dataloader, device, optimizer, criterion, caculate_func)
        
        if epoch % 10 == 0:
            print(f"Fold {fold + 1}, Epoch {epoch}: Loss = {total_loss / len(train_dataloader)}")
    
    # Evaluate the model on the test set for the current fold
    y_labels, y_probs = eval_loop(m, test_dataloader, device, caculate_func=caculate_func)
    y_prob = np.array(y_probs)
    y_true = np.array(y_labels)
    thershold = 0.4 if config.dataset else 0.25
    y_pred = np.where(y_prob > thershold, 1, 0)

    # Store the results for the current fold
    fold_results['ROC-AUC'].append(roc_auc_score(y_true, y_prob))
    fold_results['calibration'].append(brier_score_loss(y_true, y_prob))
    fold_results['precision'].append(precision_score(y_true, y_pred))
    fold_results['recall'].append(recall_score(y_true, y_pred))
    fold_results['f1'].append(f1_score(y_true, y_pred))

    y_pred = np.where(y_prob > 0.5, 1, 0)
    fold_results['precision-0.5'].append(precision_score(y_true, y_pred))
    fold_results['recall-0.5'].append(recall_score(y_true, y_pred))
    fold_results['f1-0.5'].append(f1_score(y_true, y_pred))

print(fold_results, flush=True)