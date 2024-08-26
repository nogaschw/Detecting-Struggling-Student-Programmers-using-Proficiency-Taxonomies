import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoModel, BitsAndBytesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('/home/nogaschw/Codeworkout/Thesis/Data/code_to_output_dict.pkl', 'rb') as file:
    code_output_dict = pickle.load(file)

def get_code_output(code_model, num_code, code_input_ids, code_attention_mask, mean=False):
    global code_output_dict
    global device
    snapshots = []
    for i in range(num_code):
        try:
            snapshot = code_output_dict[tuple(code_input_ids[i].tolist())]
        except:
            print("run the model, this snapshot is missing")
            with torch.no_grad():
                code_output = code_model(code_input_ids.to(device), code_attention_mask.to(device)).last_hidden_state[:, 0, :]
            for j in range(num_code):
                code_output_dict[tuple(code_input_ids[j].tolist())] = code_output[j].tolist()
            snapshot = code_output_dict[tuple(code_input_ids[i].tolist())]
        snapshots.append(snapshot)
    if mean:
        snapshot = np.mean(snapshots, axis=0)
    return torch.tensor(snapshots).to(device)


def big_language_model_support(huggingface_code_model, code_model_name):
    """
    add support in big language model require LoRa training.
    """
    print(f"use {code_model_name}")
    big_language_model = ['meta-llama/CodeLlama-7b-hf']
    if not code_model_name in big_language_model:
        return huggingface_code_model.from_pretrained(code_model_name)
    
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    
    code_model = huggingface_code_model.from_pretrained(
                    code_model_name,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
    
    lora_config = LoraConfig(
            r=2,                   # Rank of the low-rank matrices
            lora_alpha=4,          # Scaling factor for the LoRA parameters
            lora_dropout=0.1,       # Dropout rate for LoRA layers
            task_type="FEATURE_EXTRACTION"   # Task type for LoRA configuration
        )
    code_model.gradient_checkpointing_enable()
    code_model = prepare_model_for_kbit_training(code_model)
    
    return get_peft_model(code_model, lora_config)

class BasicModelMean(nn.Module):
    """
    Model that combines two models for text and code classification
    the code model make an average of the code samples
    """
    def __init__(self, text_model_name, code_model_name):
        super(BasicModelMean, self).__init__() 
        self.code_model = big_language_model_support(AutoModelForSequenceClassification, code_model_name)
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
    
    
class CodeQuestionLSTMModel(nn.Module):
    """
    Model of LSTM of the snapshots before current question + current question
    """
    def __init__(self, text_model_name, code_model_name, hidden_size, num_layers, num_classes=1, code_output_from_dict=True):
        super(CodeQuestionLSTMModel, self).__init__()
        self.code_model = big_language_model_support(AutoModel, code_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.lstm = nn.LSTM(self.code_model.config.hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.code_output_from_dict = code_output_from_dict

    def forward(self, text_input_ids, text_attention_mask, code_input_ids, code_attention_mask, questions_input_ids=None, questions_attention_mask=None, num_snapshots=0):
        batch_size, num_code, max_code_len = code_input_ids.size()

        # Reshape inputs
        code_input_ids = code_input_ids.view(batch_size * num_code, max_code_len)
        code_attention_mask = code_attention_mask.view(batch_size * num_code, max_code_len)

        # Pass through Code Model and Text Model
        with torch.no_grad():
            code_output = get_code_output(self.code_model, num_code, code_input_ids, code_attention_mask)
            text_output= self.text_model(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :]  

        # Reshape input for LSTM
        sequence_output = torch.cat((code_output, text_output), dim=0)
        sequence_output = sequence_output.view(batch_size, num_code + 1, -1)

        # Pass through LSTM
        lstm_output, _ = self.lstm(sequence_output)

        # Take the output of the last LSTM layer at the last time step
        self.lstm_last_output = lstm_output[:, -1, :]

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
        self.code_model = big_language_model_support(AutoModel, code_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.lstm = nn.LSTM(self.code_model.config.hidden_size, hidden_size, num_layers, batch_first=True)
        self.mean = mean
        self.fc = nn.Linear(hidden_size, num_classes)
        self.code_output_from_dict = code_output_from_dict

    def forward(self, text_input_ids, text_attention_mask, code_input_ids, code_attention_mask, questions_input_ids, questions_attention_mask, num_snapshots):
        batch_size, num_code, max_code_len = code_input_ids.size()
        batch_size, num_question, max_question_len = questions_input_ids.size()
        num_question_tuple = tuple([1 for i in range(num_question)])
        num_snapshots = tuple(num_snapshots.squeeze(0).tolist())

        # Reshape inputs
        code_input_ids = code_input_ids.view(batch_size * num_code, max_code_len)
        code_attention_mask = code_attention_mask.view(batch_size * num_code, max_code_len)
        questions_input_ids = questions_input_ids.view(batch_size * num_question, max_question_len)
        questions_attention_mask = questions_attention_mask.view(batch_size * num_question, max_question_len)

        # Pass through Code Model and Text Model
        with torch.no_grad():
            code_output = get_code_output(self.code_model, num_code, code_input_ids, code_attention_mask, self.mean)
            text_output= self.text_model(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :] 
            questions_outputs = self.text_model(questions_input_ids, questions_attention_mask).last_hidden_state[:, 0, :]

        # Reshape input for LSTM
        code_per_question_tensors = torch.split(code_output, num_snapshots, dim=0)
        questions = torch.split(questions_outputs, num_question_tuple, dim=0)
        final_sequence = []
        # Concatenate the tensor between the splits
        for i, t in enumerate(code_per_question_tensors):
            final_sequence.append(questions[i])
            final_sequence.append(t)
        
        final_sequence = torch.cat(final_sequence, dim=0)
        sequence_output = torch.cat((final_sequence, text_output), dim=0)
        sequence_output = sequence_output.view(batch_size, num_code + num_question + 1, -1)

        # Pass through LSTM
        lstm_output, _ = self.lstm(sequence_output)

        # Take the output of the last LSTM layer at the last time step
        self.lstm_last_output = lstm_output[:, -1, :]

        # Pass through the final linear layer
        logits = self.fc(self.lstm_last_output)
        return logits