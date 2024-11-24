import os
import copy
import wandb
import torch
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# L1 Regularization
def l1_regularization(model, lambda_l1=1e-4):
    return lambda_l1 * sum(p.abs().sum() for p in model.parameters())

# L2 Regularization (similar approach)
def l2_regularization(model, lambda_l2=1e-4):
    return lambda_l2 * sum((p ** 2).sum() for p in model.parameters())

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
        probs = torch.sigmoid(logits.cpu())
        all_probs.extend(probs)
        all_labels.extend(labels)
    all_probs = torch.cat(all_probs).tolist()
    return all_labels, all_probs


def training_loop(model, train_dataloader, test_dataloader, optimizer, criterion, device, name, weights=None):
    print(weights)
    #Initialize Variables for EarlyStopping
    best_loss = float('inf')
    best_model_weights = None
    patience = 5

    print(datetime.datetime.now().strftime('%d/%m/%Y_%H:%M:%S'), flush=True)
    print(len(train_dataloader), len(test_dataloader))
    if weights != None:
        weights = torch.tensor(weights, device=device)

    for epoch in range(100):
        print(f"Epoch: {epoch}", flush=True)
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            if i % 100 == 0:
                print(f"Batch {i} from {len(train_dataloader)}", flush=True)
            dict_batch = {k: v.to(device) for k, v in batch.items()}
            model_params = {k: v for k, v in dict_batch.items() if k != 'label'}

            outputs = model(*model_params.values())
            targets = dict_batch['label'].float()
            loss = criterion(outputs.view(-1), targets)

            if weights is not None:
                sample_weights = torch.where(targets == 1, weights[1], weights[0])
                loss = torch.mean(sample_weights * loss) 

            loss.backward()
        
            if hasattr(model, 'custom_backward'):
                model.custom_backward()  # Call custom backward if available
    
            optimizer.step()
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
                test_loss += torch.mean(loss).item()

        avg_loss_train = total_loss / len(train_dataloader)
        avg_loss_valid = test_loss / len(test_dataloader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}], LR: {current_lr:.6f}, Loss: {avg_loss_train:.4f}, Val Loss: {avg_loss_valid:.4f}, patience: {patience}', flush=True)
        wandb.log({"training loss": avg_loss_train, "avg_loss_valid loss": avg_loss_valid, "patience": patience}) 

        # Early stopping
        if avg_loss_valid < best_loss:
            best_loss = avg_loss_valid 
            best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here      
            print("success deep copy")
            patience = 5  # Reset patience counter
            torch.save(best_model_weights, name)
            print("success save in", name)
        else:
            patience -= 1
            if patience == 0:
                break
         
    torch.save(best_model_weights, name)
    print(datetime.datetime.now().strftime('%d/%m/%Y_%H:%M:%S'), flush=True)
    model.load_state_dict(best_model_weights)
    print("Loaded best model weights.")
    return model

def results(threshold, y_true, y_prob):
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    y_pred = np.where(y_prob > threshold, 1, 0)
    roc_auc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    best = "best"
    if threshold == 0.5:
        best = "0.5"
    #  df = pd.concat([pd.DataFrame([[model_name, threshold, roc_auc, accuracy, precision, recall, f1]], columns=df.columns), df], ignore_index=True)
    wandb.log({"threshold": threshold, "roc_auc": roc_auc, "accuracy": accuracy, f"precision_{best}": precision, f"recall_{best}": recall, f"f1_{best}": f1})
    cm = confusion_matrix(y_true, y_pred)
    wandb.log({f"confusion_matrix_{best}": cm})
    #  return df 