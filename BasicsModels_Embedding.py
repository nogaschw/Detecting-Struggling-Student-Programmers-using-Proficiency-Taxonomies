import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CodeQuestionLSTMModel_Embedding(nn.Module):
    """
    Model of LSTM of the snapshots before current question + current question
    """
    def __init__(self, text_model_name, hidden_size, num_layers, num_classes=1):
        super(CodeQuestionLSTMModel_Embedding, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.lstm = nn.LSTM(self.text_model.config.hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, text_input_ids, text_attention_mask, code_embedding, code_num, questions_input_ids=None, questions_attention_mask=None, num_snapshots=0, que_num=0):
        batch_size, size_code, len_code = code_embedding.size()
        with torch.no_grad():
            text_output= self.text_model(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :].unsqueeze(1)
        
        batch_indices = torch.arange(batch_size, device=code_embedding.device)
        insert_indices = code_num
        
        input_to_lstm = code_embedding.clone()        
        input_to_lstm[batch_indices, insert_indices] = text_output.long()
        
        seq_lengths = [i + 1 for i in code_num]

        packed_sequence = pack_padded_sequence(input_to_lstm, lengths=seq_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_sequence)
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        self.lstm_last_output = unpacked_output[torch.arange(batch_size), [length - 1 for length in seq_lengths], :]
        logits = self.fc(self.lstm_last_output)
        return logits
    
    
class CodeQPrevQLSTMModel_Embedding(nn.Module):
    """
    Model of LSTM that get series of: 
    q1 + snapshots of q1 + q2 + snapshots of q2 + ..... q n-1 + snapshots of qn-1 + qn + snapshots of qn + current q
    """
    def __init__(self, text_model_name, hidden_size, num_layers, num_classes=1, mean=False):
        super(CodeQPrevQLSTMModel_Embedding, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.lstm = nn.LSTM(self.text_model.config.hidden_size, hidden_size, num_layers, batch_first=True)
        self.mean = mean
        self.fc = nn.Linear(hidden_size, num_classes)
        self.embedding_cache = {}
    
    @torch.jit.script_method
    def get_text_embedding(self, input_ids, attention_mask):
        cache_key = input_ids.cpu().numpy().tobytes()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        with torch.no_grad():
            output = self.text_model(input_ids, attention_mask).last_hidden_state[:, 0, :]
        
        self.embedding_cache[cache_key] = output
        return output

    def forward(self, text_input_ids, text_attention_mask, code_embedding, code_num, questions_input_ids=None, questions_attention_mask=None, num_snapshots=0, que_num=0):
        batch_size, _, max_len = code_embedding.size()
        num_snapshots = tuple(num_snapshots.squeeze(0).tolist())

        # Pass through Code Model and Text Model
        text_output = self.get_text_embedding(text_input_ids, text_attention_mask).unsqueeze(1)

        max_seq_len = 2 * que_num.max() + 1 if self.mean else code_num.max() + que_num.max() + 1 
        input_to_lstm = torch.empty(batch_size,  max_seq_len, max_len, device=code_embedding.device)

        questions_outputs = self.get_text_embedding(questions_input_ids.view(-1, questions_input_ids.size(-1)),
                                            questions_attention_mask.view(-1, questions_attention_mask.size(-1)))

        seq_lengths = []
        # Concatenate the tensor between the splits
        for batch_idx in range(batch_size):
            for i, num_snapshot in enumerate(num_snapshots[batch_idx][:que_num[batch_idx]]):
                input_to_lstm[batch_idx, start_idx, :] = questions_outputs[batch_idx, i]
                if self.mean:
                    input_to_lstm[batch_idx, start_idx+1, :] = torch.mean(code_embedding[batch_idx, i:num_snapshot+i, :].float(), dim=0)
                    start_idx += 2
                else:
                    input_to_lstm[batch_idx, start_idx+1:start_idx+1+num_snapshot, :] = code_embedding[batch_idx, i:num_snapshot+i, :] 
                    start_idx += num_snapshot + 1
            input_to_lstm[batch_idx, start_idx, :] = text_output[batch_idx]
            seq_lengths.append(start_idx + 1)
        
        packed_sequence = pack_padded_sequence(input_to_lstm, lengths=seq_lengths, batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        packed_output, _ = self.lstm(packed_sequence)
        
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        self.lstm_last_output = unpacked_output[torch.arange(batch_size), torch.tensor(seq_lengths) - 1, :]

        # Pass through the final linear layer
        logits = self.fc(self.lstm_last_output)
        return logits
    
class CodeQuestionLSTMAttentionModel(nn.Module):
    def __init__(self, text_model_name, hidden_size, num_layers, num_classes=1):
        super(CodeQuestionLSTMAttentionModel, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.lstm = nn.LSTM(self.text_model.config.hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def attention_net(self, lstm_output, final_hidden_state):
        # Reshape final_hidden_state
        final_hidden_state = final_hidden_state.permute(1, 0, 2)  # (batch_size, num_layers * num_directions, hidden_size)
        final_hidden_state = final_hidden_state.contiguous().view(final_hidden_state.size(0), -1)  # (batch_size, num_layers * num_directions * hidden_size)
        final_hidden_state = final_hidden_state.unsqueeze(1)  # (batch_size, 1, num_layers * num_directions * hidden_size)

        # Calculate attention weights
        attn_weights = torch.tanh(self.attention(lstm_output))  # (batch_size, seq_len, 1)
        soft_attn_weights = torch.softmax(attn_weights, dim=1)  # (batch_size, seq_len, 1)

        # Calculate context vector
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).squeeze(2)  # (batch_size, hidden_size * 2)

        return context, soft_attn_weights

    def forward(self, text_input_ids, text_attention_mask, code_embedding, code_num, questions_input_ids=None, questions_attention_mask=None, num_snapshots=0, que_num=0):
        batch_size, _, _ = code_embedding.size()
        with torch.no_grad():
            text_output = self.text_model(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :]
        
        batch_indices = torch.arange(batch_size, device=code_embedding.device)
        insert_indices = code_num
        
        input_to_lstm = code_embedding.clone()        
        input_to_lstm[batch_indices, insert_indices] = text_output.long()
        
        seq_lengths = [i + 1 for i in code_num]

        packed_sequence = pack_padded_sequence(input_to_lstm, lengths=seq_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_sequence.float())

        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        attn_output, _ = self.attention_net(lstm_output, hidden)
        logits = self.fc(attn_output)
        return logits
