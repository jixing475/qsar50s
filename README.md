# QSAR50S: Machine Learning-Based Virtual Screening for 50S Ribosomal Inhibitors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub stars](https://img.shields.io/github/stars/jixing475/qsar50s?style=social)](https://github.com/jixing475/qsar50s)
[![GitHub forks](https://img.shields.io/github/forks/jixing475/qsar50s?style=social)](https://github.com/jixing475/qsar50s)

A comprehensive Python package for machine learning-based virtual screening of 50S ribosomal inhibitors targeting *Cutibacterium acnes*. This package implements the complete workflow from data preprocessing to QSAR model development and virtual screening, as described in our published research.

## 🎯 Features

- **Data Preprocessing**: Automated ChEMBL data curation and activity labeling
- **Molecular Feature Generation**: RDKit and PaDEL-Descriptor integration for fingerprints and descriptors
- **QSAR Modeling**: Implementation of ANN and Random Forest models with hyperparameter tuning
- **Virtual Screening**: Complete pipeline for compound screening and prioritization
- **ADMET Filtering**: Integrated pharmacokinetic property prediction
- **Reproducible Research**: Command-line tools and Python API for reproducible workflows
- **Model Interpretability**: Feature importance analysis and visualization support

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [Project Structure](#project-structure)
- [Models and Methods](#models-and-methods)
- [Performance Metrics](#performance-metrics)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Java Runtime Environment (JRE) 8 or higher (for PaDEL-Descriptor)

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/jixing475/qsar50s.git
cd qsar50s

# Create conda environment
conda env create -f environment.yml
conda activate qsar50s

# Install the package in development mode
pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/jixing475/qsar50s.git
cd qsar50s

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Post-Installation

After installation, verify that everything is working:

```bash
# Test basic functionality
python -c "import qsar50s; print('QSAR50S installed successfully!')"

# Check PaDEL-Descriptor installation
python -c "from padelpy import padeldescriptor; print('PaDEL-Descriptor is available')"
```

## 🏃‍♂️ Quick Start

Here's a quick example to get you started with the complete workflow:

```bash
# 1. Preprocess ChEMBL data
python scripts/01_data_preprocessing.py \
    --input data/raw/chembl_ribosomal_inhibitors.csv \
    --output data/processed/training_data.csv

# 2. Generate molecular fingerprints
python scripts/02_feature_generation.py \
    --input data/processed/training_data.csv \
    --output data/processed \
    --fingerprint-type MACCS \
    --method rdkit

# 3. Train QSAR model
python scripts/03_model_training.py \
    --input data/processed/data_fp_MACCS_train.csv \
    --output models/ \
    --model-type RF \
    --tune \
    --remove-outliers \
    --save-plots
```

## 💻 Usage

### Command Line Interface

The package provides several command-line scripts for each step of the workflow:

#### Data Preprocessing

```bash
# Process ChEMBL data with default settings
python scripts/01_data_preprocessing.py \
    -i data/raw/chembl_data.csv \
    -o data/processed/training_data.csv

# With custom configuration and plotting
python scripts/01_data_preprocessing.py \
    -i data/raw/chembl_data.csv \
    -o data/processed/training_data.csv \
    -c config/custom_config.yaml \
    --plot
```

#### Feature Generation

```bash
# Generate PubChem fingerprints using PaDEL-Descriptor
python scripts/02_feature_generation.py \
    -i data/processed/training_data.csv \
    -o data/processed \
    -f PubChem \
    -m padel \
    -d train

# Generate MACCS fingerprints using RDKit
python scripts/02_feature_generation.py \
    -i data/processed/training_data.csv \
    -o data/processed \
    -f MACCS \
    -m rdkit \
    --generate-descriptors

# Process SMI file directly
python scripts/02_feature_generation.py \
    --smi-input data/external/compound_library.smi \
    -o data/processed \
    -f PubChem
```

#### Model Training

```bash
# Train Random Forest model with hyperparameter tuning
python scripts/03_model_training.py \
    -i data/processed/data_fp_PubChem_train.csv \
    -o models/ \
    -m RF \
    --tune \
    --remove-outliers \
    --save-plots

# Train ANN model
python scripts/03_model_training.py \
    -i data/processed/data_fp_MACCS_train.csv \
    -o models/ \
    -m ANN \
    --test-size 0.25
```

### Python API

You can also use the package programmatically:

```python
import pandas as pd
from qsar50s.data.preprocessing import DataProcessor
from qsar50s.features.fingerprints import FingerprintGenerator
from qsar50s.models.qsar_models import RFModel

# Initialize processors
data_processor = DataProcessor('config/paths.yaml')
fp_generator = FingerprintGenerator('config/paths.yaml')
model = RFModel('config/paths.yaml')

# 1. Preprocess data
df_processed = data_processor.process_chembl_data(
    'data/raw/chembl_data.csv',
    'data/processed/training_data.csv'
)

# 2. Generate fingerprints
X, y = model.load_data('data/processed/training_data.csv')
X_fp = fp_generator.generate_rdkit_fingerprints(X, 'Morgan')

# 3. Train model
model.build_model(n_estimators=200, max_depth=20)
performance = model.train_model(X_fp, y, hyperparameter_tuning=True)

# 4. Make predictions
predictions = model.predict(X_fp)

# 5. Save model
model.save_model('models/rf_model.pkl')
```

## 📁 Project Structure

```
qsar50s/
├── qsar50s/                    # Main Python package
│   ├── __init__.py
│   ├── data/                   # Data preprocessing module
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── features/              # Feature generation module
│   │   ├── __init__.py
│   │   └── fingerprints.py
│   ├── models/                # QSAR modeling module
│   │   ├── __init__.py
│   │   └── qsar_models.py
│   ├── screening/             # Virtual screening module
│   │   ├── __init__.py
│   │   └── virtual_screening.py
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── rdkit_utils.py
│       └── molecular_utils.py
├── scripts/                   # Command-line scripts
│   ├── 01_data_preprocessing.py
│   ├── 02_feature_generation.py
│   └── 03_model_training.py
├── data/                      # Data files
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data
│   └── external/              # External tools (PaDEL-Descriptor)
├── config/                    # Configuration files
│   └── paths.yaml
├── models/                    # Trained models
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── examples/                  # Example notebooks and scripts
├── requirements.txt           # Python dependencies
├── environment.yml           # Conda environment
└── README.md                 # This file
```

## 🔬 Models and Methods

### Data Preprocessing
- **ChEMBL Data Curation**: Automated filtering and cleaning of ChEMBL bioactivity data
- **Activity Labeling**: pMIC calculation and activity threshold classification (pMIC ≥ 6 for active compounds)
- **Quality Control**: Duplicate removal and outlier detection

### Molecular Feature Generation
- **RDKit Fingerprints**: Morgan, MACCS, and RDKit topological fingerprints
- **PaDEL-Descriptor Integration**: PubChem, EState, CDK, and other fingerprint types
- **Molecular Descriptors**: Physicochemical property calculation
- **Automated Processing**: Support for both CSV and SMI file formats

### QSAR Modeling
- **Artificial Neural Networks (ANN)**: Multi-layer perceptron with hyperparameter optimization
- **Random Forest (RF)**: Ensemble method with feature importance analysis
- **Model Validation**: 5-fold cross-validation and external test set evaluation
- **Feature Selection**: Variance threshold and correlation-based filtering

### Key Features
- **PCA-based Outlier Detection**: Identifies and removes problematic samples
- **Hyperparameter Tuning**: Grid search optimization for model parameters
- **Comprehensive Evaluation**: Multiple metrics (RMSE, MAE, R², CV scores)
- **Model Interpretability**: Feature importance and visualization support
- **Reproducible Workflows**: Configuration-based pipeline management

## 📝 Citation

If you use this package in your research, please cite our paper:

```bibtex
xxx
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate and follow the existing code style.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/jixing475/qsar50s.git
cd qsar50s

# Create development environment
conda env create -f environment.yml
conda activate qsar50s

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black qsar50s/ scripts/
```

## 🙏 Acknowledgments

- ChEMBL database for bioactivity data
- RDKit and PaDEL-Descriptor developers
- The Python scientific computing community


---

**Note**: This package is intended for research purposes only. Please validate predictions experimentally before drawing biological conclusions.