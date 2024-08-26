import copy
import torch
import pickle
import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def eval_loop(model, test_dataloader, device):
    all_labels = []
    all_probs = []
    for i, batch in enumerate(test_dataloader):
        if i % 1000 == 0:
            print(f"Batch {i} from {len(test_dataloader)}", flush=True)
        dict_batch = {k: v.to(device) for k, v in batch.items()}
        model_params = {k: v for k, v in dict_batch.items() if k != 'label'}
        with torch.no_grad():
            logits = model(*model_params.values()) 
        labels = dict_batch['label'].cpu().numpy()
        probs = torch.sigmoid(logits).cpu()
        all_probs.extend(probs)
        all_labels.extend(labels)
    all_probs = torch.cat(all_probs).tolist()
    return all_labels, all_probs


def training_loop(model, train_dataloader, test_dataloader, optimizer, criterion, device, lr_scheduler, name):
    #Initialize Variables for EarlyStopping
    best_loss = float('inf')
    best_model_weights = None
    patience = 5
    checkpoint_dir = "/home/nogaschw/Codeworkout/Thesis/Models"
    checkpoint_path = f"{checkpoint_dir}/{name}.pth"

    print(datetime.datetime.now().strftime('%d/%m/%Y_%H:%M:%S'), flush=True)
    print(len(train_dataloader), len(test_dataloader))

    for epoch in range(100):
        print(f"Epoch: {epoch}", flush=True)
        model.train(True)
        total_loss = 0

        for i, batch in enumerate(train_dataloader):
            if i % 1000 == 0:
                print(f"Batch {i} from {len(train_dataloader)}", flush=True)
            dict_batch = {k: v.to(device) for k, v in batch.items()}
            model_params = {k: v for k, v in dict_batch.items() if k != 'label'}

            outputs = model(*model_params.values())
            loss = criterion(outputs.squeeze(1), dict_batch['label'].float())
        
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if i % 1000 == 0:
                    print(f"Test Batch {i} from {len(test_dataloader)}", flush=True)
                dict_batch = {k: v.to(device) for k, v in batch.items()}
                model_params = {k: v for k, v in dict_batch.items() if k != 'label'}
                logits = model(*model_params.values()) 

                loss = criterion(logits.squeeze(1), dict_batch['label'].float())
                test_loss += loss.item()

        avg_loss_train = total_loss / len(train_dataloader)
        avg_loss_valid = test_loss / len(test_dataloader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}], LR: {current_lr:.6f}, Loss: {avg_loss_train:.4f}, Val Loss: {avg_loss_valid:.4f}, patience: {patience}', flush=True)

        # Early stopping
        if avg_loss_valid < best_loss:
            best_loss = avg_loss_valid            
            torch.save(model.state_dict(), checkpoint_path)
            best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here      
            patience = 5  # Reset patience counter
        else:
            patience -= 1
            if patience == 0:
                break
        if current_lr < 0.0000001:
            break
        
    print(datetime.datetime.now().strftime('%d/%m/%Y_%H:%M:%S'), flush=True)


def results(df, model_name, y_prob, y_true):
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    y_pred = np.round(y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    df = pd.concat([pd.DataFrame([[model_name, roc_auc, accuracy, precision, recall, f1]], columns=df.columns), df], ignore_index=True)
    return df
        


def create_code_output(code_model, train_dataloader , valid_dataloader, test_dataloader, device):
    rows = {}
    def _to_each_data(dataloader):
        for i, batch in tqdm(enumerate(dataloader)):
            if i % 1000 == 0:
                print(f"Batch {i} from {len(dataloader)}")
            batch_size, num_code, max_code_len = batch['code_input_ids'].size()
            
            code_input_ids = batch['code_input_ids'].view(batch_size * num_code, max_code_len)
            code_attention_mask = batch['code_attention_mask'].view(batch_size * num_code, max_code_len)
            
            with torch.no_grad():
                # Pass data through the model
                code_output = code_model(code_input_ids.to(device), code_attention_mask.to(device)).last_hidden_state[:, 0, :]

            # Collect rows
            for j in range(batch_size * num_code):
                rows[tuple(code_input_ids[j].tolist())] = code_output[j].tolist()
    _to_each_data(train_dataloader)
    print(f"Finish train- dict size {len(rows)}")
    _to_each_data(valid_dataloader)
    print(f"Finish valid- dict size {len(rows)}")
    _to_each_data(test_dataloader)
    print(f"Finish test- dict size {len(rows)}")
    # Save the dictionary to a file
    with open('Data/code_to_output_dict.pkl', 'wb') as file:
        pickle.dump(rows, file)

    