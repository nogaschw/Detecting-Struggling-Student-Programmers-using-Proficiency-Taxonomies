import torch
from helper import *
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from transformers import AutoModel
from torch.nn.utils.rnn import pack_padded_sequence

class BasicModel(nn.Module):
    def __init__(self, text_model_name, hidden_size_lstm, input_lstm, size_text_mlp, dropout_rate=0.2):
        super(BasicModel, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
       
        self.lstm_snapshots = nn.LSTM(
            self.text_model.config.hidden_size, 
            input_lstm, 
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_lstm_output_prev_labels = nn.BatchNorm1d(input_lstm + 1) # +1 for prev_struggling

        self.lstm_submissions = nn.LSTM(
            input_lstm + 1,  # +1 for prev_struggling
            hidden_size_lstm,
            batch_first=True,
            bidirectional=False,
        )

        self.MLP_text = self._make_mlp(size_text_mlp, hidden_size_lstm, dropout_rate)
        self.batch_norm_code_mlp = nn.BatchNorm1d(hidden_size_lstm + 3) 
        self.MLP_code = self._make_mlp(hidden_size_lstm + 3, hidden_size_lstm, dropout_rate) # +3 to stats
        self.fully_connected_layer = nn.Linear(hidden_size_lstm, 1)
    
    def _make_mlp(self, input_dim, hidden_dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def _compute_attempt_statistics(self, code_num):
        code_num = code_num.float()
        num_attempts = code_num.sum(dim=1, keepdim=True)
        median_attempts = code_num.median(dim=1, keepdim=True).values
        max_attempts = torch.max(code_num, dim=-1, keepdim=True).values
        return torch.cat([num_attempts, max_attempts, median_attempts], dim=-1)
    
    def _question_similarity(self, prev_q_ids, prev_q_mask, text_output, batch_size, q_num):
        questions_similarity = torch.zeros((batch_size, q_num), dtype=torch.float)
        for i in range(q_num):
            prev_q_embedding = self._process_text_input(prev_q_ids[:, i], prev_q_mask[:, i])
            # questions_similarity[:, i] = F.cosine_similarity(text_output, prev_q_embedding, dim=-1)
            pearson_similarity, _ = pearsonr(text_output.cpu().detach().numpy().flatten(), prev_q_embedding.cpu().detach().flatten())
            questions_similarity[:, i] = pearson_similarity
        return questions_similarity.unsqueeze(-1).to(prev_q_ids.device)
    
    def _process_code_snapshots(self, code_embedding, code_num, batch_size, q_num, c_padding, len):
        # Pack and process code snapshots
        snapshots_lstm = pack_padded_sequence(
            code_embedding.view(batch_size * q_num, c_padding, len),
            lengths=code_num.view(batch_size * q_num).to('cpu'),
            batch_first=True,
            enforce_sorted=False
        )
        
        _, (snapshots_h, _) = self.lstm_snapshots(snapshots_lstm)
        snapshots_h = self.dropout(snapshots_h)
        return snapshots_h.view(batch_size, q_num, -1)
        
    def _process_submissions_sequence(self, snapshots_output, prev_struggling, batch_size, q_num, question_similarity=None):
        # Prepare and process submissions sequence
        if question_similarity != None:
            snapshots_output = snapshots_output * question_similarity
        input_lstm_submissions = self.norm_lstm_output_prev_labels(
            torch.cat((snapshots_output, prev_struggling.float().unsqueeze(-1)), dim=-1).view(batch_size * q_num, -1)
        ).view(batch_size, q_num, -1)
        
        _, (final_submission_h, _) = self.lstm_submissions(input_lstm_submissions)
        final_submission_h = self.dropout(final_submission_h)
        return final_submission_h.view(batch_size, -1)
    
    def _process_text_input(self, text_input_ids, text_attention_mask):
        with torch.no_grad():
            text_output = self.text_model(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :]
        return text_output
    
    def _generate_prediction(self, text_mlp, submissions_output, stats, difficulty_skill=None):
        input_code_mlp = self.batch_norm_code_mlp(torch.cat((stats, submissions_output), dim=-1))
        code_output = self.MLP_code(input_code_mlp)
        text_output = self.MLP_text(text_mlp)
        # Combine text and code representations
        combined =  text_output + code_output
        if not difficulty_skill is None:
            combined = self.batch_norm_all(torch.cat((combined, difficulty_skill), dim=-1))
        return self.fully_connected_layer(combined)
    
    def _backward(self):
        # List all affected layers
        affected_parameters = {
            'batch_norm_code_mlp': (self.batch_norm_code_mlp.weight, F.pad(self.mask, (3, 0), value=1)),
            'MLP_code_layer0': (self.MLP_code[0].weight, F.pad(self.mask, (3, 0), value=1)),
            'MLP_code_layer1': (self.MLP_code[1].weight, self.mask),
            'MLP_code_layer4': (self.MLP_code[4].weight, self.mask),
            'MLP_code_layer5': (self.MLP_code[5].weight, self.mask),
            'fully_connected_layer': (self.fully_connected_layer.weight, self.mask)
        }

        # Filter gradients for all affected layers
        with torch.no_grad():
            for name, (param, func) in affected_parameters.items():
                if param.grad is not None: # Apply the Q-matrix mask to the gradients
                    grad_mask = func
                    if param.grad.dim() > 1:
                        grad_mask = grad_mask.T if param.grad.size(0) != grad_mask.size(0) else grad_mask
                    param.grad *= grad_mask.expand_as(param.grad) # Apply the mask
                    param -= param.grad # Apply gradient descent step
                    if name == 'KCweight': # For positivity constraint, apply ReLU-like operation to KCweight
                        param.data = torch.maximum(param.data, torch.zeros_like(param.data))

            # Normalize affected layers for stability
            for name, (param, func) in affected_parameters.items():
                if param.grad is not None: # Normalize weights (if applicable) to ensure stability
                    norm_factor = param.data.norm(dim=0, keepdim=True) if param.data.dim() > 1 else param.data.sum()
                    param.data /= (norm_factor + 1e-5)  # Prevent division by zero
    
class AllWithLastSnapshot(BasicModel):
    def __init__(self, text_model_name):
        super(AllWithLastSnapshot, self).__init__(text_model_name, hidden_size_lstm=512, input_lstm=768,  size_text_mlp=768)
        
    def forward(self, text_input_ids, text_attention_mask, code_embedding, prev_q_ids, prev_q_mask, code_num, prev_struggling): 
        batch_size, q_num, c_padding, len = code_embedding.size()
        text_output = self._process_text_input(text_input_ids, text_attention_mask)
        question_similarity = self._question_similarity(prev_q_ids, prev_q_mask, text_output, batch_size, q_num)
        final_submission_h = self._process_submissions_sequence(code_embedding.squeeze(2), prev_struggling, batch_size, q_num, question_similarity)
        output =  self._generate_prediction(text_output, final_submission_h, self._compute_attempt_statistics(code_num))
        return output
    
class AllWithLastSnapshotTax(BasicModel):
    def __init__(self, text_model_name, num_feature=10):
        super(AllWithLastSnapshotTax, self).__init__(text_model_name, hidden_size_lstm=512, input_lstm=768,  size_text_mlp=768)
        self.before_Qmatrix_layer = nn.Linear(512, num_feature)
        self.MLP_text = self._make_mlp(768, 10, 0.2)
        self.batch_norm_code_mlp = nn.BatchNorm1d(10 + 3)
        self.MLP_code = self._make_mlp(10 + 3, 10, 0.2)
        self.fully_connected_layer = nn.Linear(10, 1)
        
    def forward(self, text_input_ids, text_attention_mask, code_embedding, prev_q_ids, prev_q_mask, code_num, prev_struggling, features): 
        batch_size, q_num, c_padding, len = code_embedding.size()
        text_output = self._process_text_input(text_input_ids, text_attention_mask)
        question_similarity = self._question_similarity(prev_q_ids, prev_q_mask, text_output, batch_size, q_num)
        final_submission_h = self._process_submissions_sequence(code_embedding.squeeze(2), prev_struggling, batch_size, q_num, question_similarity)
        # integrate q matrix
        final_submission_h_with_q_matrix = self.before_Qmatrix_layer(final_submission_h)
        final_submission_h_with_q_matrix = torch.sigmoid(final_submission_h_with_q_matrix) * features
        # Pass final_submission_h_with_q_matrix to the rest of the network
        output = self._generate_prediction(text_output, final_submission_h_with_q_matrix, self._compute_attempt_statistics(code_num))
        # save q_matrix_for_backprop
        self.mask = features.sum(dim=0) / features.sum()
        return output
    
    def custom_backward(self):
        self._backward()

    
class All(BasicModel):
    def __init__(self, text_model_name):
        super(All, self).__init__(text_model_name, hidden_size_lstm=512, input_lstm=512,  size_text_mlp=768)
        
    def forward(self, text_input_ids, text_attention_mask, code_embedding, prev_q_ids, prev_q_mask, code_num, prev_struggling): 
        batch_size, q_num, c_padding, len = code_embedding.size()
        text_output = self._process_text_input(text_input_ids, text_attention_mask)
        question_similarity = self._question_similarity(prev_q_ids, prev_q_mask, text_output, batch_size, q_num)
        code_num = code_num.float()
        snapshots_h = self._process_code_snapshots(code_embedding, code_num, batch_size, q_num, c_padding, len)
        final_submission_h = self._process_submissions_sequence(snapshots_h, prev_struggling, batch_size, q_num, question_similarity)
        output = self._generate_prediction(text_output, final_submission_h, self._compute_attempt_statistics(code_num))
        return output
    
class AllWithTaxonomy(BasicModel):
    def __init__(self, text_model_name, num_feature=10):
        super(AllWithTaxonomy, self).__init__(text_model_name, hidden_size_lstm=512, input_lstm=512,  size_text_mlp=768 + num_feature)
        self.batch_norm_text_mlp = nn.BatchNorm1d(768 + num_feature)
        
    def forward(self, text_input_ids, text_attention_mask, code_embedding, prev_q_ids, prev_q_mask, code_num, prev_struggling, features): 
        batch_size, q_num, c_padding, len = code_embedding.size()
        text_output = self._process_text_input(text_input_ids, text_attention_mask)
        question_similarity = self._question_similarity(prev_q_ids, prev_q_mask, text_output, batch_size, q_num)
        code_num = code_num.float()
        snapshots_h = self._process_code_snapshots(code_embedding, code_num, batch_size, q_num, c_padding, len)
        final_submission_h = self._process_submissions_sequence(snapshots_h, prev_struggling, batch_size, q_num, question_similarity)
        input_text_mlp = self.batch_norm_text_mlp(torch.cat((features, text_output), dim=-1))
        output = self._generate_prediction(input_text_mlp, final_submission_h, self._compute_attempt_statistics(code_num))
        return output
    
class AllWithTaxonomyQmatrix(BasicModel):
    def __init__(self, text_model_name, num_feature=10, dropout_rate=0.2):
        super(AllWithTaxonomyQmatrix, self).__init__(text_model_name, hidden_size_lstm=512, input_lstm=512,  size_text_mlp=768, dropout_rate=dropout_rate)
        self.before_Qmatrix_layer = nn.Linear(512, num_feature)
        self.MLP_text = self._make_mlp(768, 10, dropout_rate)
        self.batch_norm_code_mlp = nn.BatchNorm1d(10 + 3)
        self.MLP_code = self._make_mlp(10 + 3, 10, dropout_rate)
        self.fully_connected_layer = nn.Linear(10, 1)

    def forward(self, text_input_ids, text_attention_mask, code_embedding, prev_q_ids, prev_q_mask, code_num, prev_struggling, features): 
        batch_size, q_num, c_padding, len = code_embedding.size()
        text_output = self._process_text_input(text_input_ids, text_attention_mask)
        question_similarity = self._question_similarity(prev_q_ids, prev_q_mask, text_output, batch_size, q_num)
        code_num = code_num.float()
        snapshots_h = self._process_code_snapshots(code_embedding, code_num, batch_size, q_num, c_padding, len)
        final_submission_h = self._process_submissions_sequence(snapshots_h, prev_struggling, batch_size, q_num, question_similarity)
        # q matrix
        final_submission_h_with_q_matrix = self.before_Qmatrix_layer(final_submission_h)
        final_submission_h_with_q_matrix = torch.sigmoid(final_submission_h_with_q_matrix) * features
        # Pass final_submission_h_with_q_matrix to the rest of the network
        output = self._generate_prediction(text_output, final_submission_h_with_q_matrix, self._compute_attempt_statistics(code_num))
        # save q_matrix_for_backprop
        self.mask = features.sum(dim=0) / features.sum()
        return output
              
    def custom_backward(self):
        self._backward()

class AllWithTaxonomyPrevQmatrix(BasicModel):
    def __init__(self, text_model_name, num_feature=10):
        super(AllWithTaxonomyPrevQmatrix, self).__init__(text_model_name, hidden_size_lstm=512, input_lstm=512, size_text_mlp=768)
        self.num_feature = num_feature
        self.before_Qmatrix_layer = nn.Linear(512 + 1, self.num_feature)
        self.KCweight = nn.Linear(num_feature * 30, num_feature * 30, bias=False)
        self.MLP_text = self._make_mlp(768, 300, 0.2)
        self.batch_norm_code_mlp = nn.BatchNorm1d(300 + 3)
        self.MLP_code = self._make_mlp(300 + 3, 300, 0.2)
        self.fully_connected_layer = nn.Linear(300, 1)

    def forward(self, text_input_ids, text_attention_mask, code_embedding, prev_q_ids, prev_q_mask, code_num, prev_struggling, last_features): 
        batch_size, q_num, c_padding, len = code_embedding.size()
        text_output = self._process_text_input(text_input_ids, text_attention_mask)
        code_num = code_num.float()
        snapshots_h = self._process_code_snapshots(code_embedding, code_num, batch_size, q_num, c_padding, len)
        question_similarity = self._question_similarity(prev_q_ids, prev_q_mask, text_output, batch_size, q_num)
        snapshots_h = self.norm_lstm_output_prev_labels(
            torch.cat((snapshots_h, prev_struggling.float().unsqueeze(-1)), dim=-1).view(batch_size * q_num, -1)
        ).view(batch_size, q_num, -1) * question_similarity
        # integrate q matrix
        before_q_matrix = self.before_Qmatrix_layer(snapshots_h)
        question_with_q_matrix = (before_q_matrix * last_features).view(batch_size, q_num * self.num_feature)
        # Multiply with KC weights after Q matrix is applied
        question_with_q_matrix = self.KCweight(question_with_q_matrix)
        # Pass final_submission_h_with_q_matrix to the rest of the network
        output = self._generate_prediction(text_output, question_with_q_matrix, self._compute_attempt_statistics(code_num))
        # save q_matrix_for_backprop
        self.mask = last_features.view(batch_size, q_num * self.num_feature).max(dim=0)[0]
        return output
        
    def custom_backward(self):
        self._backward()