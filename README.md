This repository contains the code for my Master's thesis, including the code of the ECAI paper and the extended version used for the thesis and journal.

## Environment Setup

### Creating a New Environment

To set up the environment for this project, follow these steps:

1. **Create a virtual environment:**
   ```bash
   python -m venv thesis_env
   source thesis_env/bin/activate  # On Windows: thesis_env\Scripts\activate
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **For GPU support (optional):**
   Ensure you have CUDA installed and compatible PyTorch version. The requirements.txt includes GPU-compatible versions.

## Preprocessing

The `Preprocessing` folder contains scripts to preprocess the data before use:

- **`Preprocessing_Codeworkout.py`**: Preprocesses the CodeWorkout dataset
- **`Preprocessing_Falcon.py`**: Preprocesses the Falcon dataset  
- **`CreateDataEmbedding.py`**: Creates embeddings for code snippets using pre-trained models
- **`Config.py`**: Configuration settings for preprocessing

These scripts handle data cleaning, feature extraction, and prepare the datasets in the required format for model training.

## Configuration

The `Config.py` file contains different configurations for the model and experiments:

- **Dataset selection**: Choose between CodeWorkout (0), Falcon (1), or Both (2)
- **Model variants**: Switch between PTM model and ablation studies
- **Gemma integration**: Enable/disable cognitive skills integration using Gemma/QWAN models
- **Hyperparameters**: Learning rate, batch size, epochs, hidden dimensions, etc.

Key configuration options:
- `self.dataset`: Controls which dataset to use
- `self.run_model`: Specifies model type ('PTM' for main model)
- `self.gemma_path`: Path to cognitive skills data (None to disable)

## Experiment Tracking with WandB

This project uses **Weights & Biases (WandB)** for experiment tracking and visualization. WandB logs training metrics, validation losses, hyperparameters, and evaluation results to help monitor and compare different experimental runs.

### Logged Metrics
- Training and validation losses
- Binary classification and skill prediction losses
- Learning rates and patience counters
- Evaluation metrics (ROC-AUC, accuracy, precision, recall, F1-score)
- Confusion matrices

### Disabling WandB
If you prefer not to use WandB tracking, you can disable it by setting `use_wandb=False` when calling the training functions:

```python
# In your training script
model = training_loop(
    model=model,
    train_dataloader=train_dataloader, 
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    name=model_name,
    use_wandb=False  # Disable WandB logging
)

# For evaluation
results(threshold=0.5, y_true=y_true, y_prob=y_prob, use_wandb=False)
```

When disabled, all metrics will still be printed to the console but won't be logged to WandB.

## Model

The `Model` folder contains the core implementation with several components:

### 1. Baselines
Implementation of baseline models for comparison:
- **DKT (Deep Knowledge Tracing)**
- **CodeDKT**: Specialized DKT for code
- **SAKT**: Self-Attentive Knowledge Tracing

### 2. Models
The main models directly connected to the configuration:
- **PTM (Programming Task Model)**: Our main contribution
- **Ablation studies**: Various model variants for analysis

### 3. Run Options

- **`run_k_fold_merge.py`**: Runs both datasets together in k-fold cross-validation
- **`run_k_fold.py`**: Runs the model specified in config with k-fold validation
- **`run_optuna.py`**: Hyperparameter optimization using Optuna (customizable search space)
- **`run_train_val_test.py`**: Standard train/validation/test split with early stopping (supports pre-defined split IDs)

## Programming Concepts

The `programming_concepts.json` file contains the final programming concepts and details on how concepts are merged in each dataset.

## Cognitive Skills

The cognitive skills folder contains the extended version of the ECAI paper. This component:

1. **Agent-based Assessment**: Uses AI agents (Gemma or QWAN) to estimate students' cognitive skills based on code snapshots
2. **Cognitive Skills Evaluated**:
   - **Decomposition**: Breaking problems into manageable parts
   - **Algorithmic Design**: Structuring and choosing appropriate algorithms  
   - **Reading Comprehension**: Understanding problem requirements

3. **Integration**: Skills are integrated into the main model by setting `self.gemma_path` in the config file to point to the generated cognitive skills data

### Usage
To use cognitive skills integration:
1. Run the appropriate Gemma/QWAN scripts to generate cognitive assessments
2. Update `self.gemma_path` in `Config.py` to point to the generated CSV files
3. Run your chosen model script - the cognitive skills will be automatically incorporated

## Model Architecture

The PTM (Programming Task Model) architecture integrates:
- Text encoding for problem descriptions
- Code sequence modeling with LSTM
- Attention mechanisms for skill importance
- Latent student representations
- Cognitive skills integration (when enabled)
