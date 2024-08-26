
import torch
import torch.nn as nn
from BasicsModels import *
from sklearn.preprocessing import StandardScaler

class ModelBasicCases(nn.Module):
        """
        Model combine as feature the score of the students- how many test he finish.
        The combine will be with attention layer.
        """
        
        
class ModelComputationalConstructs(nn.Module):
        """
        Model combine as feature the computational constructs ​need to be used in the question.
        If the computational construct require it will be 0 or 1 depend if.
        """
        def __init__(self, base_model, base_model_num_layer, computational_constructs_dim, output_dim=1):
                super(ModelComputationalConstructs, self).__init__()
                self.base_model = base_model 
                self.final_linear = nn.Linear(computational_constructs_dim + base_model_num_layer, output_dim)
                self.lstm_normalizer = nn.LayerNorm(base_model_num_layer)
                self.features_normalizer = nn.LayerNorm(computational_constructs_dim)
                
        def forward(self, text_input_ids, text_attention_mask, code_input_ids, code_attention_mask,
                        questions_input_ids, questions_attention_mask, num_snapshots, features):

                self.base_model(text_input_ids, text_attention_mask, code_input_ids, code_attention_mask, questions_input_ids, questions_attention_mask, num_snapshots)
                lstm_output = self.base_model.lstm_last_output

                # Normalize LSTM output and features separately
                normalized_features = self.features_normalizer(features)
                normalized_lstm_output = self.lstm_normalizer(lstm_output)
                
                # Combine normalized outputs
                combined_input = torch.cat([normalized_lstm_output, normalized_features], dim=-1)
                
                # Final linear layer
                final_output = self.final_linear(combined_input)
                return final_output
