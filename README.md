# Multi-Modal Model Code

1. A bidirectional cross-attention machine learning framework for medical image classification.
2. A robust imputation and classification pipeline for tabular datasets.

## Overview

This repository contains two primary analysis pipelines:
1. **Transformer-based multi-modal classification** - Combines medical images with tabular clinical data
2. **Tabular-only classification** - Traditional ML models with feature selection and imputation analysis

## Installation

### Prerequisites
- Python 3 with conda
- CUDA-capable GPU (recommended for transformer training)

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Create conda environment
conda env create -f conda_env.yml
conda activate <env-name>
