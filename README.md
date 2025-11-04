# Multi-Modal Model Code

1. A bidirectional cross-attention machine learning framework for medical image classification.
2. A robust imputation and classification pipeline for tabular datasets.

### Prerequisites
- Python 3 with conda
- CUDA-capable GPU (recommended for transformer training)

## General Overview

This repository contains two primary analysis pipelines:
1. **Transformer-based multi-modal classification** - Combines medical images with tabular clinical data
2. **Tabular-only classification** - Traditional ML models with feature selection and imputation analysis

## Overview of tabular-dataset pipeline functions

### feature_selection.py
`select_features()` implements a rigorous, publication-quality feature selection pipeline that combines:

- **Model-agnostic importance**: SHAP values work across all model types
- **Nested cross-validation**: Prevents overfitting in feature selection
- **Ensemble stability**: Features must be consistently important across iterations
- **Correlation filtering**: Removes redundant features automatically

`select_model()` identifies the optimal classifier and imputation method combination for your dataset through exhaustive cross-validated evaluation. This function implements **Step 2 (Model Selection)** from the paper's methodology and should be run **after feature selection**.

`train_and_predict()` is a comprehensive machine learning pipeline function that handles the complete workflow for binary classification tasks, including:

- Missing value imputation
- Automatic class imbalance handling
- Hyperparameter optimization via RandomizedSearchCV
- Model training and evaluation

### mm_analysis.py
`analyze_downstream_imputations()` provides comprehensive evaluation of imputation methods by measuring both **imputation quality** (how accurate are the imputed values) and **downstream performance** (how well does classification work with imputed data).

### impute_data.py
`custom_impute_df()` provides a unified interface for 9 different missing value imputation strategies, from simple statistical methods to advanced machine learning approaches. Designed for flexibility in research and production ML pipelines.

## Overview of transformer pipeline functions

### transformer_training.py
`main()` implements a complete deep learning pipeline for medical image classification using multimodal data (chest X-rays + lab values). Built on Vision Transformers with cross-attention fusion for combining imaging and tabular features.

### Key Features

- **Multimodal Fusion**: Combines chest X-ray images with 42 lab measurements
- **Class Balance**: Intelligent resampling to match data distributions
- **Multiple Training Modes**: Standard training, evaluation, hyperparameter search, fine-tuning
- **Robust Validation**: Bootstrapped estimates for confidence intervals
- **Flexible Configuration**: Easy experimentation through configuration variables


# Create conda environment
conda env create -f conda_env.yml
conda activate <env-name>
