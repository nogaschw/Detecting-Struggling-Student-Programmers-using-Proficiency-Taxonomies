import os
import sys
import torch
import optuna
from helper import *
from Models import *
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
sys.path.append(os.path.join(os.getcwd(), 'Codeworkout/Thesis/Data'))
from Data import *
from choosedataset import *
sys.path.append(os.path.join(os.getcwd(), 'Codeworkout/Thesis'))
from Config import *
from sklearn.metrics import roc_curve
from transformers import AutoTokenizer
import Data_Embedding as Data_Embedding
from torch.nn.utils.rnn import pack_padded_sequence


config = Config()

class TBPP_Generation(nn.Module):
    def __init__(self, text_model_name, code_len, num_skills=10,
                 snapshots_hl=512, submission_hl=512, dropout_rate=0.3):
        """
        Initialize the TBPP_Generation model.

        Parameters:
        text_model_name (str): Name of the pre-trained text model.
        code_len (int): Length of the code sequences.
        num_skills (int): Number of skills.
        snapshots_hl (int): Hidden size for LSTM on code snapshots.
        submission_hl (int): Hidden size for LSTM on task submissions.
        dropout_rate (float): Dropout rate.
        """
        super(TBPP_Generation, self).__init__()
        self.snapshots_hl = snapshots_hl
        self.submission_hl = submission_hl
        self.dropout_rate = dropout_rate

        self.text_model = AutoModel.from_pretrained(text_model_name)

        self.lstm_snapshots = nn.LSTM(
            code_len,
            self.snapshots_hl,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm_lstm_output_prev_labels = nn.BatchNorm1d(self.snapshots_hl + 1)

        self.lstm_submissions = nn.LSTM(
            self.snapshots_hl + 1,
            self.submission_hl,
            batch_first=True,
            bidirectional=False
        )

        self.skills_rep = nn.Linear(self.submission_hl, num_skills)

    def _represent_task_submissions(self, code_embedding, prev_struggling, code_num):
        if (code_num == 1).all():
            snapshots_output = code_embedding.squeeze(2)
        else:
            batch_size, q_num, c_padding, length = code_embedding.size()
            snapshots_lstm = pack_padded_sequence(
                code_embedding.view(batch_size * q_num, c_padding, length),
                lengths=code_num.view(batch_size * q_num).to('cpu'),
                batch_first=True,
                enforce_sorted=False
            )
            _, (snapshots_h, _) = self.lstm_snapshots(snapshots_lstm)
            snapshots_output = self.dropout(snapshots_h).view(batch_size, q_num, -1)

        batch_size, q_num, _ = snapshots_output.size()
        task_submissions = self.norm_lstm_output_prev_labels(
            torch.cat((snapshots_output, prev_struggling.float().unsqueeze(-1)), dim=-1).view(batch_size * q_num, -1)
        ).view(batch_size, q_num, -1)
        return task_submissions

    def _process_submissions_sequence(self, task_submissions, task_num):
        batch_size, q_num, length = task_submissions.size()
        task_submissions_lstm = pack_padded_sequence(
            task_submissions,
            lengths=task_num.to('cpu'),
            batch_first=True,
            enforce_sorted=False
        )
        _, (final_submission_h, _) = self.lstm_submissions(task_submissions_lstm)
        final_submission_h = self.dropout(final_submission_h)
        return self.skills_rep(final_submission_h.view(batch_size, -1))

    def _process_text_input(self, text_input_ids, text_attention_mask):
        with torch.no_grad():
            text_output = self.text_model(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :]
        return text_output

    def createTBPP(self, text_input_ids, text_attention_mask, code_embedding, code_num, task_num, prev_struggling):
        task_submissions = self._represent_task_submissions(code_embedding, prev_struggling, code_num)
        text_output = self._process_text_input(text_input_ids, text_attention_mask)
        TBPP = self._process_submissions_sequence(task_submissions, task_num)
        return text_output, TBPP

class PTM(TBPP_Generation):
    def __init__(self, text_model_name, code_len, num_students, num_skills, num_prog_concepts,
                output_mlp_builder, num_attention_heads, snapshots_hl, submission_hl):
        super().__init__(text_model_name, code_len, num_skills=num_prog_concepts)

        input_dim = self.text_model.config.hidden_size
        self.latent_matrix = nn.Embedding(num_students, 3)
        self.fc_latent = nn.Linear(num_prog_concepts, 3)
        self.fc_interaction = nn.Linear(6, 3)

        self.skill_query = nn.Linear(num_skills, input_dim)
        self.skill_key = nn.Linear(num_skills, input_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_attention_heads, dropout=0.1)

        self.skill_importance = nn.Sequential(
            nn.Linear(num_skills, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

        self.output_mlp = output_mlp_builder(input_dim, num_skills)
 
    def _struggling_prediction(self, text_embedding, skill_vector, required_prog_concepts):
        """
        Predict struggling using cross attention and skill importance.
        
        Parameters:
        text_embedding (torch.Tensor): Text embeddings.
        skill_vector (torch.Tensor): Skill vector.
        required_prog_concepts (torch.Tensor): Required programming concepts.
        
        Returns:
        torch.Tensor: Struggling prediction.
        """
        batch_size = skill_vector.size(0)
        required_skills = torch.cat((required_prog_concepts, torch.ones((batch_size, 3)).to(required_prog_concepts.device)), dim=1)
        skill_weights = self.skill_importance(required_skills)# Skill importance weighting
        # Perform cross attention
        attended_features, _ = self.cross_attention(self.skill_query(skill_vector).unsqueeze(0), self.skill_key(skill_vector).unsqueeze(0), text_embedding.unsqueeze(0))
        # Skill-weighted feature transformation
        return self.output_mlp(torch.cat([attended_features.squeeze(0), skill_vector * skill_weights], dim=-1))
    

    def latent_skills_represention(self, student_id, skill_vec):
        """
        Represent latent skills using student ID and skill vector.
        
        Parameters:
        student_id (torch.Tensor): Student IDs.
        skill_vec (torch.Tensor): Skill vector.
        
        Returns:
        torch.Tensor: Latent skills representation.
        """
        latent_features_id = self.latent_matrix(student_id.long())  # Get latent features from embedding
        latent_features_skills = self.fc_latent(skill_vec)  # Latent features from skills
        latent_features = torch.cat((latent_features_id, latent_features_skills), dim=1)
        return self.fc_interaction(latent_features)

    def forward(self, text_input_ids, text_attention_mask, code_embedding, code_num, task_num, prev_struggling, required_prog_concepts, student_id): 
        """
        Forward pass of the PTM model.
        
        Parameters:
        text_input_ids (torch.Tensor): Target task text input IDs.
        text_attention_mask (torch.Tensor): Target task text attention mask.
        code_embedding (torch.Tensor): Code embeddings after LLM.
        code_num (torch.Tensor): Number of code sequences.
        prev_struggling (torch.Tensor): Previous struggling labels.
        required_prog_concepts (torch.Tensor): Required programming concepts.
        student_id (torch.Tensor): Student IDs.
        
        Returns:
        tuple: Model output and TBPP representation.
        """
        text_output, skill_vec = self.createTBPP(text_input_ids, text_attention_mask, code_embedding, code_num, task_num, prev_struggling)
        latent_skills = self.latent_skills_represention(student_id, skill_vec)
        TBPP = torch.cat((skill_vec, latent_skills), dim=1) # concate with known skills
        output = self._struggling_prediction(text_output, TBPP, required_prog_concepts)
        return output, torch.sigmoid(TBPP)

def build_output_mlp(input_dim, num_skills, config):
    layers = []
    in_dim = input_dim + num_skills

    if config.use_layernorm:
        layers.append(nn.LayerNorm(in_dim))

    activation_fn = nn.ReLU() if config.activation == 'relu' else nn.GELU()

    for hidden_dim in config.mlp_hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation_fn)
        in_dim = hidden_dim

    layers.append(nn.Dropout(config.dropout_rate))
    layers.append(nn.Linear(in_dim, 1))

    return nn.Sequential(*layers)

def train_with_config(config):
    latents = True if config.run_model == 'PTM' and config.gemma_path is None else False
    data = [Codeworkout, Falcon, Both][config.dataset](latents=latents, similarity=False, gemma_path=config.gemma_path)
    df = data.df

    Data_Embedding.load(data.embedded)
    df['prev_tasks'] = df['prev_tasks'].apply(lambda x: [i[-config.padding_size_code:] for i in x])

    text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Data_Embedding.DatasetPTM
    model = PTM(
        config.text_model_name,
        config.max_len_code,
        len(np.unique(df['student_id_encoded'])),
        13,
        10,
        output_mlp_builder=lambda input_dim, num_skills: build_output_mlp(input_dim, num_skills, config),
        num_attention_heads=config.num_attention_heads,
        snapshots_hl=config.snapshots_hl,
        submission_hl=config.submission_hl
    )
    caculate_func = caculate_2losses
    criterion = [nn.BCEWithLogitsLoss(), nn.L1Loss()]

    train_loader, val_loader, test_loader = create_data_loader(
        df, dataset, text_tokenizer,
        max_len_code=config.max_len_code,
        padding_size_q=config.number_coding_tasks,
        padding_size_code=config.padding_size_code,
        batch_size=config.batch_size,
        create_split=False,
        ids_filepath_prefix=f'/home/nogaschw/Codeworkout/UseData/both/'
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    model = model.to(device)

    model = training_loop(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        name=f'/tmp/tmp_model.pth',
        caculate_func=caculate_func,
        use_wandb=False
    )

    all_labels, all_probs = eval_loop(model, val_loader, device, caculate_func=caculate_func)
    val_auc = roc_auc_score(all_labels, all_probs)
    return 1.0 - val_auc

def objective(trial):
    config = Config()
    config.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    config.dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    config.snapshots_hl = trial.suggest_categorical("snapshots_hl", [128, 256, 512])
    config.submission_hl = trial.suggest_categorical("submission_hl", [128, 256, 512])
    config.num_attention_heads = trial.suggest_categorical("num_attention_heads", [2, 4, 6, 8])

    # MLP layers config
    config.mlp_hidden_dims = trial.suggest_categorical("mlp_hidden_dims", [[128, 64], [256, 64], [64], [128, 64, 32]])
    config.use_layernorm = trial.suggest_categorical("use_layernorm", [True, False])
    config.activation = trial.suggest_categorical("activation", ['relu', 'gelu'])

    config.config_name = f"OptunaTrial_{trial.number}"
    print(f"Trial {trial.number} params: {config.__dict__}")

    return train_with_config(config)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("Best config:", study.best_params)