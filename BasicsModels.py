import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModelForSequenceClassification, AutoModel, BitsAndBytesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# with open('/home/nogaschw/Codeworkout/Thesis/Data/merged_dict.pkl', 'rb') as file:
#     code_output_dict = pickle.load(file)
# print(len(code_output_dict))

def get_code_output(code_model, num_code, code_input_ids, code_attention_mask, text_output,mean=False):
    global code_output_dict
    global device

    batch_size, max_code_len, _ = code_input_ids.size()
    num_code = [min(length, max_code_len) for length in num_code]

    if mean:
        num_code = [1] * batch_size

    max_len = max(num_code)
    snapshots = torch.zeros((batch_size, max_len + 1, text_output.size(1)), dtype=torch.float32, device=device)

    for batch_idx in range(batch_size):
        batch_snapshots = []
        input_ids = code_input_ids[batch_idx, :num_code[batch_idx]].cpu().tolist()
        
        for i, input_id in enumerate(input_ids):
            input_id_tuple = tuple(input_id)
            if input_id_tuple in code_output_dict:
                snapshot = torch.tensor(code_output_dict[input_id_tuple], dtype=torch.float32, device=device)
            else:
                print("Run the model, this snapshot is missing")
                with torch.no_grad():
                    code_output = code_model(code_input_ids[batch_idx, i].unsqueeze(0).to(device),
                                            code_attention_mask[batch_idx, i].unsqueeze(0).to(device)).last_hidden_state[:, 0, :]
                    snapshot = code_output.squeeze(0)
                    code_output_dict[input_id_tuple] = snapshot.cpu().tolist()
            batch_snapshots.append(snapshot)
        batch_snapshots = torch.stack(batch_snapshots)
        if mean:
            batch_snapshots = batch_snapshots.mean(dim=0, keepdim=True)

        snapshots[batch_idx, :num_code[batch_idx]] = batch_snapshots
        snapshots[batch_idx, num_code[batch_idx]] = text_output[batch_idx]

    seq_lengths = [length + 1 for length in num_code]
    return snapshots, seq_lengths

class BasicModelMean(nn.Module):
    """
    Model that combines two models for text and code classification
    the code model make an average of the code samples
    """
    def __init__(self, text_model_name, code_model_name):
        super(BasicModelMean, self).__init__() 
        self.code_model = AutoModelForSequenceClassification.from_pretrained(code_model_name)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(text_model_name) 
        self.text_scale = nn.Parameter(torch.tensor(1.0))
        self.code_scale = nn.Parameter(torch.tensor(1.0))
        self.linear = nn.Linear(4, 1)
        
    def forward(self, text_input_ids, text_attention_mask, code_input_ids, code_attention_mask):   
        text_output = self.text_model(input_ids=text_input_ids, attention_mask=text_attention_mask)

        code_output = self.code_model(input_ids=code_input_ids, attention_mask=code_attention_mask)

        logits_text = text_output.logits
        logits_code = code_output.logits

        logits_text = logits_text * self.text_scale
        logits_code = logits_code * self.code_scale

        combined_logits = torch.cat([logits_text, logits_code], dim=-1)
        logits = self.linear(combined_logits)
            
        return logits
    
class FullyConnectedNetwork(nn.Module):
    def __init__(self, code_model_name, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNetwork, self).__init__()
        self.code_model = AutoModelForSequenceClassification.from_pretrained(code_model_name)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, code_input_ids, code_attention_mask):
        batch_size, num_question, max_len_code = code_input_ids.size()
        batch_code_input_ids_reshaped = code_input_ids.view(-1, max_len_code)
        batch_code_attention_mask_reshaped = code_attention_mask.view(-1, max_len_code)
        
        with torch.no_grad():
            code_model_output = self.code_model(batch_code_input_ids_reshaped, batch_code_attention_mask_reshaped).logits
        reshaped_logits = code_model_output.view(batch_size, -1)

        x = self.fc1(reshaped_logits)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

class QuestionLSTMModel(nn.Module):
    def __init__(self, text_model_name, hidden_size, num_layers, num_classes=1):
        super(QuestionLSTMModel, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.lstm = nn.LSTM(self.text_model.config.hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Cache for text embeddings
        self.embedding_cache = {}

    def forward(self, text_input_ids, text_attention_mask, code_input_ids=None, code_attention_mask=None, code_num=None, questions_input_ids=None, questions_attention_mask=None, num_snapshots=0, que_num=0):        
        # Check cache for text embeddings
        cache_key = text_input_ids.cpu().numpy().tobytes()
        if cache_key in self.embedding_cache:
            text_output = self.embedding_cache[cache_key]
        else:
            with torch.no_grad():
                text_output = self.text_model(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :].unsqueeze(1)
            self.embedding_cache[cache_key] = text_output
        
        lstm_output, _ = self.lstm(text_output)
        
        self.lstm_last_output = lstm_output[:, -1, :]
        
        logits = self.fc(self.lstm_last_output)
        return logits

    @torch.jit.export
    def get_embedding(self, text_input_ids, text_attention_mask):
        with torch.no_grad():
            return self.text_model(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :]
    

class CodeQuestionLSTMModel(nn.Module):
    """
    Model of LSTM of the snapshots before current question + current question
    """
    def __init__(self, text_model_name, code_model_name, hidden_size, num_layers, num_classes=1, code_output_from_dict=True):
        super(CodeQuestionLSTMModel, self).__init__()
        self.code_model = AutoModel.from_pretrained(code_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.lstm = nn.LSTM(self.code_model.config.hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.code_output_from_dict = code_output_from_dict

    def forward(self, text_input_ids, text_attention_mask, code_input_ids, code_attention_mask, code_num, questions_input_ids=None, questions_attention_mask=None, num_snapshots=0, que_num=0):
        # Pass through Code Model and Text Model
        with torch.no_grad():
            text_output= self.text_model(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :]  
            sequence_output, seq_lengths = get_code_output(self.code_model, code_num, code_input_ids, code_attention_mask, text_output)
        
        packed_sequence = pack_padded_sequence(sequence_output, lengths=seq_lengths, batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        packed_output, _ = self.lstm(packed_sequence)
        
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        final_outputs = []
        for i, length in enumerate(seq_lengths):
            final_outputs.append(unpacked_output[i, length - 1, :])  # Extract last valid output per sample
        
        self.lstm_last_output = torch.stack(final_outputs) 

        # Pass through the final linear layer
        logits = self.fc(self.lstm_last_output)
        return logits
    
class CodeQPrevQLSTMModel(nn.Module):
    """
    Model of LSTM that get series of: 
    q1 + snapshots of q1 + q2 + snapshots of q2 + ..... q n-1 + snapshots of qn-1 + qn + snapshots of qn + current q
    """
    def __init__(self, text_model_name, code_model_name, hidden_size, num_layers, num_classes=1, code_output_from_dict=True, mean=False):
        super(CodeQPrevQLSTMModel, self).__init__()
        self.code_model = AutoModel.from_pretrained(code_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.lstm = nn.LSTM(self.code_model.config.hidden_size, hidden_size, num_layers, batch_first=True)
        self.mean = mean
        self.fc = nn.Linear(hidden_size, num_classes)
        self.code_output_from_dict = code_output_from_dict

    def forward(self, text_input_ids, text_attention_mask, code_input_ids, code_attention_mask, code_num, questions_input_ids=None, questions_attention_mask=None, num_snapshots=0, que_num=0):
        batch_size = code_input_ids.size(0)
        batch_size, num_question, max_question_len = questions_input_ids.size()
        num_snapshots = tuple(num_snapshots.squeeze(0).tolist())

        # Pass through Code Model and Text Model
        with torch.no_grad():
            text_output= self.text_model(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :] 
            sequence_output, seq_lengths = get_code_output(self.code_model, code_num, code_input_ids, code_attention_mask, text_output, self.mean)
        
        max_seq_len = sequence_output.size(1) + que_num.max().item()
        snapshots_tensor = torch.zeros((batch_size, max_seq_len, text_output.size(1)), dtype=torch.float32, device=device)
        
        # Concatenate the tensor between the splits
        for batch_idx in range(batch_size):
            with torch.no_grad():
                questions_outputs = self.text_model(questions_input_ids[batch_idx][:que_num[batch_idx]], questions_attention_mask[batch_idx][:que_num[batch_idx]]).last_hidden_state[:, 0, :]
            q = torch.cat((questions_outputs[0].unsqueeze(0), sequence_output[batch_idx]), dim=0)
            snapnum = 0
            for inx,i in enumerate(num_snapshots[batch_idx]):
                snapnum += i + 1
                if inx < int(que_num[batch_idx]) - 1:
                    q = torch.cat(( q[:snapnum], questions_outputs[inx + 1].unsqueeze(0), q[snapnum:]))
            snapshots_tensor[batch_idx, :q.size(0)] = q
        
        seq_lengths = [int(que_num[inx]) + i for inx,i in enumerate(seq_lengths)]
        packed_sequence = pack_padded_sequence(snapshots_tensor, lengths=seq_lengths, batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        packed_output, _ = self.lstm(packed_sequence)
        
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        self.lstm_last_output = unpacked_output[torch.arange(batch_size), [length - 1 for length in seq_lengths], :]

        # Pass through the final linear layer
        logits = self.fc(self.lstm_last_output)
        return logits