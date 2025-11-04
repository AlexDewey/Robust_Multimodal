import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import shuffle, resample
import torchvision.transforms as transforms
import pickle
import os
import random

from missingness_functions import calculate_tab_completeness, sparse_remove_data_until_percentage_complete, remove_rows_until_complete
from impute_data import custom_impute_df

from transformer_models.Better_Token_Concat_Transformer import MMViTTransformer
from transformer_models.Matmul_Transformer import MatMul_Transformer
from transformer_models.Cross_Attention_Transformer import MMViTCrossAttentionTransformer
from transformer_models.Cross_Attention_Transformer_2 import MMViTCrossAttentionTransformer_2
from transformer_models.Bert_Cross_Attention_Transformer import MMViTCrossAttentionBERTTransformer

def patient_stratified_split(df, test_size, random_state, disease):
    # Create df of unique subject_ids with most common label
    subject_labels = df.groupby('subject_id')[disease].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()

    # Get unique subject_ids and split into stratifying labels
    unique_subject_ids = subject_labels['subject_id'].values
    train_subject_ids, test_subject_ids = train_test_split(
        unique_subject_ids, 
        test_size=test_size,
        stratify=subject_labels[disease],
        random_state=random_state
    )

    # Create train and test dataframes
    train_df = df[df['subject_id'].isin(train_subject_ids)]
    test_df = df[df['subject_id'].isin(test_subject_ids)]

    return train_df, test_df

class MultiModalDataset(Dataset):
    """
    Dataset class that handles both image and tabular data with varying levels of missingness.
    """
    def __init__(self, df, lab_data_columns, model, disease, split):
        self.df = df
        self.lab_data_columns = lab_data_columns
        self.split = split
        self.disease = disease
        
        # Get data transforms from timm
        data_config = timm.data.resolve_model_data_config(model)
        self.transforms = timm.data.create_transform(**data_config, is_training=(split == 'train'))

        train_df, temp_df = patient_stratified_split(df, test_size=0.2, random_state=1, disease=disease)
        val_df, test_df = patient_stratified_split(temp_df, test_size=0.5, random_state=1, disease=disease)

        if split == 'train':
            self.transforms = transforms.Compose([
                self.transforms,
                transforms.RandomRotation(degrees=[-5, 5])
            ])
            self.df = train_df.reset_index(drop=True)
        elif split == 'val':
            self.df = val_df.reset_index(drop=True)
        elif split == 'test':
            self.df = test_df.reset_index(drop=True)

        self.df[lab_data_columns] = (self.df[lab_data_columns] - self.df[lab_data_columns].min()) / (self.df[lab_data_columns].max() - self.df[lab_data_columns].min() + 1e-8) # Avoiding div 0 for nan values
        # Impute missing values
        
        self.df[lab_data_columns] = custom_impute_df(self.df[lab_data_columns], 'mice-dt')
        # Normalize lab data
        self.df[lab_data_columns] = (self.df[lab_data_columns] - self.df[lab_data_columns].min()) / (self.df[lab_data_columns].max() - self.df[lab_data_columns].min() + 1e-8) # Avoiding div 0 for nan values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image
        img_path = self.df.iloc[idx]['file_location']
        image = Image.open(img_path)
        image = self.transforms(image)
        
        # First ensure the values are numeric type
        tabular_data = self.df.iloc[idx][self.lab_data_columns].values
        tabular_data = tabular_data.astype(np.float32)  # Convert to float32
        tabular = torch.tensor(tabular_data, dtype=torch.float32)
        
        # Get label
        label = torch.tensor(self.df.iloc[idx][self.disease], dtype=torch.float32)
        
        return {
            'image': image,
            'tabular': tabular,
            'label': label
        }

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        images = batch['image'].to(device)
        tabular = batch['tabular'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images, tabular)
        loss = criterion(outputs, labels.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    
    return total_loss / len(dataloader), auc, auprc

import matplotlib.pyplot as plt

def plot_binned_results(binned_results, step=5, max_bin=40):
    """
    Plot binned results aggregated every `step` bins, up to `max_bin`.
    Green = correct, Red = incorrect.
    """
    bin_ranges = [(i, i+step-1) for i in range(0, max_bin+1, step)]
    labels = [f"{r[0]}-{r[1]}" for r in bin_ranges]
    
    correct_counts = []
    incorrect_counts = []
    
    for start, end in bin_ranges:
        correct = sum(binned_results[i]["correct"] for i in range(start, end+1) if i in binned_results)
        incorrect = sum(binned_results[i]["incorrect"] for i in range(start, end+1) if i in binned_results)
        correct_counts.append(correct)
        incorrect_counts.append(incorrect)
    
    x = np.arange(len(labels))
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, correct_counts, color="green", label="Correct")
    plt.bar(x, incorrect_counts, bottom=correct_counts, color="red", label="Incorrect")
    
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Number of zeros in tabular features (grouped by 5)")
    plt.ylabel("Count")
    plt.title("Prediction correctness by missingness bins")
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate(model, dataloader, criterion, device, lab_data, max_bin=40):
    """Evaluate the model and record performance binned by number of zeros in lab data."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            tabular = batch['tabular'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, tabular).squeeze(1)  # (batch,)
            preds = (outputs > 0.5).long()  # binary classification
            
            # Move to CPU for counting
            labels_cpu = labels.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            tabular_cpu = batch['tabular'].cpu().numpy()

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    return auc, auprc

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, y_disease, experiment_name, fine_tuning):
    
    best_val_auc = 0
    patience = 0
    max_patience = 2
    
    if fine_tuning != "none":
        # Load pre-trained weights
        model.load_state_dict(torch.load('./data/model_weights/' + str(fine_tuning) + '_' + str(y_disease) + '.pt'))

    for epoch in range(num_epochs):
        # Train
        train_loss, train_auc, train_auprc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_auc, val_auprc = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'AUC: {train_auc:.4f}, AUPRC: {train_auprc:.4f}')
        print(f'AUC: {val_auc:.4f}, AUPRC: {val_auprc:.4f}')
        
        # Early stopping
        if val_auc > best_val_auc:
            patience = 0
            best_val_auc = val_auc
            # Save best model
            torch.save(model.state_dict(), './data/model_weights/' + str(experiment_name) + '_' + str(y_disease) + '.pt')
        else:
            patience += 1
            if patience >= max_patience:
                print('Early stopping triggered')
                break
            
    return model, best_val_auc

def add_file_path(df):
    valid_rows = []
    for idx, img_row in df.iterrows():
        subject_id = str(img_row['subject_id'])
        directory_id = str(img_row['subject_id'])[:2]
        study_id = str(img_row['study_id'])
        dicom_id = str(img_row['dicom_id'])

        file_path = f"./data/image/physionet.org/files/mimic-cxr-jpg/2.1.0/files/p{directory_id}/p{subject_id}/s{study_id}/{dicom_id}.jpg"

        if os.path.exists(file_path):
            img_row['file_location'] = file_path
            valid_rows.append(img_row)

    reduced_pa_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    return reduced_pa_df

def main(threshold_setting: float,
         seed: int = 0) -> float:
    """
    Train and evaluate a multimodal Vision Transformer for medical image classification.
    
    Implements a complete deep learning pipeline for chest X-ray disease classification
    using both imaging (CXR) and tabular (lab values) data. Supports multiple experimental
    modes including training, evaluation, hyperparameter search, and fine-tuning.
    
    The function handles class imbalance through data filtering and resampling,
    ensuring the incomplete data distribution matches the complete data distribution. Supports
    bootstrapped validation for robust performance estimation.

    Returns
    -------
    float
        Best validation AUROC achieved during hyperparameter search (only when param_search=True).
        Returns None for training/evaluation modes.

    Configuration Variables (Edit in Function Body)
    ------------------------------------------------
    data_training : str, default="none"
        Training data mode:
        - "incomplete": Train on incomplete data (rows below completeness threshold)
        - "training": Train on complete data split
        - "none": Skip training, load pre-trained weights for evaluation
    
    data_testing : str, default="validation"
        Evaluation data split:
        - "validation": Evaluate on validation split with bootstrapping
        - "test": Evaluate on held-out test split
    
    experiment_name : str, default="cross_att_small_tokens_transformer_finetune_MICE-DT_finetune"
        Unique identifier for saving/loading model weights.
        Format: {architecture}_{imputation}_{finetuning}
        Weights saved to: ./data/model_weights/{experiment_name}_{disease}.pt
    
    fine_tuning_base : str, default="none"
        Base experiment name for transfer learning:
        - "none": Train from scratch
        - "{experiment_name}": Load weights from specified experiment for fine-tuning
    
    param_search : bool, default=False
        Hyperparameter search mode:
        - False: Standard training/evaluation with fixed hyperparameters
        - True: Random search over hyperparameter grid (10 iterations)
    
    best_disease_dict : dict
        Disease-specific configuration mapping disease names to [threshold, imputation, model]:
        - threshold (float): Completeness threshold for this disease
        - imputation (str): Not currently used (legacy parameter)
        - model (str): Not currently used (legacy parameter)
        Example: {"No_Finding": [0.65, 'mean', 'BT']}
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Can be incomplete, training, or "none"
    data_training = "none"
    # Can be validation or test
    data_testing = "validation"
    # Name models for experiment, can be "none"
    experiment_name = "cross_att_small_tokens_transformer_finetune_MICE-DT_finetune"
    # Name fine-tuning if possible, can be "none" or the name of the experiment
    fine_tuning_base = "none"
    # Are we just looking for best hyperparameters?
    param_search = False

    best_disease_dict = { # 65
        "No_Finding": [threshold_setting, 'mean', 'BT'],
    }

    lab_data = ['Glucose-Mean', 'Oxygen_Saturation-Mean', 'Temperature-Mean', 'pH-Mean',
                    'White_Blood_Cells-Mean', 'C_Reactive_Proteins-Mean', 'Diastolic_Blood_Pressure-Mean',
                    'Fraction_Inspired_Oxygen-Mean', 'Heart_Rate-Mean', 'Body_Height-Mean',
                    'Mean_Blood_Pressure-Mean', 'Respiratory_Rate-Mean', 'Systolic_Blood_Pressure-Mean',
                    'Body_Weight-Mean', 'Glucose-Var', 'Oxygen_Saturation-Var', 'Temperature-Var',
                    'pH-Var', 'White_Blood_Cells-Var', 'C_Reactive_Proteins-Var',
                    'Diastolic_Blood_Pressure-Var', 'Fraction_Inspired_Oxygen-Var', 'Heart_Rate-Var',
                    'Body_Height-Var', 'Mean_Blood_Pressure-Var', 'Respiratory_Rate-Var',
                    'Systolic_Blood_Pressure-Var', 'Body_Weight-Var', 'Glucose-Last',
                    'Oxygen_Saturation-Last', 'Temperature-Last', 'pH-Last', 'White_Blood_Cells-Last',
                    'C_Reactive_Proteins-Last', 'Diastolic_Blood_Pressure-Last',
                    'Fraction_Inspired_Oxygen-Last', 'Heart_Rate-Last', 'Body_Height-Last',
                    'Mean_Blood_Pressure-Last', 'Respiratory_Rate-Last', 'Systolic_Blood_Pressure-Last',
                    'Body_Weight-Last']


    for y_disease in best_disease_dict:

        conditions = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged_Cardiomediastinum", "Fracture", 
                "Lung_Lesion", "Lung_Opacity", "No_Finding", "Pleural_Effusion", "Pleural_Other", "Pneumothorax", 
                "Pneumonia", "Support_Devices"]
        with open(f"./data/tabular/pa_df.pkl", "rb") as f:
            pa_df = pickle.load(f)
        conditions.remove(y_disease)
        pa_df.drop(columns=conditions, inplace=True)
        data = pa_df.drop(columns=['StudyDate'])

        data_copy = data.copy()
        complete_data = remove_rows_until_complete(data=data_copy, percentage=best_disease_dict[y_disease][0])
        complete_data = add_file_path(complete_data)

        # Incomplete data are all rows in data that aren't in complete_data
        incomplete_data = data_copy[~data_copy.index.isin(complete_data.index)]
        incomplete_data = add_file_path(incomplete_data)

        pos_ratio = complete_data[y_disease].mean()

        # Split incomplete data by label
        incomplete_pos = incomplete_data[incomplete_data[y_disease] == 1]
        incomplete_neg = incomplete_data[incomplete_data[y_disease] == 0]

        # Calculate current ratio
        curr_pos = len(incomplete_pos)
        curr_neg = len(incomplete_neg)
        curr_ratio = curr_pos / (curr_pos + curr_neg)

        # Desired ratio from complete data: pos_ratio = pos / (pos + neg)
        # Solve: new_pos / new_neg = pos_ratio / (1 - pos_ratio)
        desired_ratio = pos_ratio / (1 - pos_ratio)

        if curr_ratio > pos_ratio:
            # Too many positives relative to negatives - oversample negatives
            new_neg_count = int(curr_pos / desired_ratio)
            incomplete_neg = resample(incomplete_neg, replace=True, n_samples=new_neg_count, random_state=seed)
            
        elif curr_ratio < pos_ratio:
            # Too many negatives relative to positives - oversample positives
            new_pos_count = int(desired_ratio * curr_neg)
            incomplete_pos = resample(incomplete_pos, replace=True, n_samples=new_pos_count, random_state=seed)

        # Combine the resampled data
        incomplete_data = pd.concat([incomplete_pos, incomplete_neg]).sample(frac=1, random_state=seed)

        if not param_search:
            # Initialize model
            model = MMViTCrossAttentionTransformer_2(num_tabular_features=len(lab_data),
                                        num_transformer_layers=2,
                                        num_heads=1,
                                        hidden_dim=8,
                                        dropout=0.5)
            model = model.to(device)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(),
                                        lr=0.001,
                                        weight_decay=0)

            # Model experiment setup
            if data_training == "incomplete":
                train_dataset = MultiModalDataset(df=incomplete_data, lab_data_columns=lab_data, 
                                            model=model.encoder, disease=y_disease, split='train')
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            elif data_training == "training":
                train_dataset = MultiModalDataset(df=complete_data, lab_data_columns=lab_data, 
                                            model=model.encoder, disease=y_disease, split='train')
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            if data_testing == "validation":
                val_dataset = MultiModalDataset(df=complete_data, lab_data_columns=lab_data, 
                                        model=model.encoder, disease=y_disease, split='val')
                
                boot_df = resample(
                    val_dataset.df,
                    replace=True,
                    n_samples=len(val_dataset.df),
                    random_state=seed,
                    stratify=val_dataset.df['No_Finding']
                )

                val_dataset.df = boot_df

                print(val_dataset.df['No_Finding'].value_counts())
            elif data_testing == "test":
                val_dataset = MultiModalDataset(df=complete_data, lab_data_columns=lab_data, 
                                        model=model.encoder, disease=y_disease, split='test')

            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

            if data_training == "none":
                print(f"Evaluating model for {y_disease}...")
                model.load_state_dict(torch.load('./data/model_weights/' + str(experiment_name) + '_' + str(y_disease) + '.pt'))
                val_auroc, val_auprc = evaluate(
                    model, val_loader, criterion, device, lab_data=lab_data
                )
                print(f'AUC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}')
            else:
                print(f"Training model for {y_disease}...")
                model, auc = train_model(model, train_loader, val_loader, criterion, 
                                optimizer, num_epochs=10, device=device, y_disease=y_disease,
                                experiment_name=experiment_name, fine_tuning=fine_tuning_base)
        
        else:
            # RANDOM GRID SEARCH
            param_grid = {
                'num_transformer_layers': [1, 2, 4],
                'num_heads': [1, 2, 4, 8],
                'dropout': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
                'learning_rate': [0.0001, 0.001, 0.01, 0.1],
                'weight_decay': [0.0, 0.001, 0.01, 0.1],
                'batch_size': [32, 64, 128],
                'hidden_dim': [8, 16, 32] #768
             }

            best_auc = 0
            best_params = None

            for i in range(10):

                params = {
                    key: random.choice(value) 
                    for key, value in param_grid.items()
                }

                # Initialize model
                model = MMViTCrossAttentionTransformer_2(num_tabular_features=len(lab_data),
                                        num_transformer_layers=params['num_transformer_layers'],
                                        num_heads=params['num_heads'],
                                        hidden_dim=params['hidden_dim'],
                                        dropout=params['dropout'])
                model = model.to(device)
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(),
                                            lr=params['learning_rate'],
                                            weight_decay=params['weight_decay'])

                # Setup datasets and dataloaders
                train_dataset = MultiModalDataset(df=complete_data, lab_data_columns=lab_data, 
                                                model=model.encoder, disease=y_disease, split='train')
                val_dataset = MultiModalDataset(df=complete_data, lab_data_columns=lab_data, 
                                            model=model.encoder, disease=y_disease, split='val')
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True)
                
                print(f"Training model for {y_disease}...")
                print(params)

                model, auc = train_model(model, train_loader, val_loader, criterion, 
                                optimizer, num_epochs=1, device=device, y_disease=y_disease,
                                experiment_name=experiment_name, fine_tuning=fine_tuning_base)
                
                if best_auc < auc:
                    best_auc = auc
                    best_params = params
            
            print(f"Best params for {y_disease}: {best_params}")
            print(f"Best AUC for {y_disease}: {best_auc}")

            return auc

thresholds = [0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 
              0.80, 0.825, 0.85, 0.875, 0.90, 0.925]

results = {thr: [] for thr in thresholds}

# Run 20 times per threshold
for threshold_setting in thresholds:
    print(f"Running threshold {threshold_setting}...")
    for run in range(10):
        print(f"  Run {run+1}/10")
        auc = main(threshold_setting=threshold_setting, seed=run) 
        results[threshold_setting].append(auc)

# Convert to mean and std
means = [np.mean(results[t]) for t in thresholds]
stds = [np.std(results[t]) for t in thresholds]

# Plot
plt.figure(figsize=(10, 6))
means = np.array(means)
stds = np.array(stds)
thresholds = np.array(thresholds)

plt.plot(thresholds, means, label="Mean AUROC", color="blue")
plt.fill_between(thresholds, means - stds, means + stds, color="blue", alpha=0.2, label="±1 Std Dev")

plt.xlabel("Threshold Setting")
plt.ylabel("AUROC")
plt.title("AUROC vs Threshold (20 runs each)")
plt.legend()
plt.grid(True)
plt.show()
