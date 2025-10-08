"""
QSAR50S: A Python package for machine learning-based virtual screening of 50S ribosomal inhibitors.

This package provides tools for:
- Data preprocessing and curation
- Molecular feature calculation and fingerprinting
- ML-QSAR model development and validation
- Virtual screening workflows
- ADMET prediction

Author: Jixing Liu
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Jixing Liu"

from .data import preprocessing
from .features import fingerprints, descriptors
from .models import ann_model, rf_model, evaluation
from .screening import virtual_screening, admet
from .utils import rdkit_utils, molecular_utils

__all__ = [
    'preprocessing',
    'fingerprints', 
    'descriptors',
    'ann_model',
    'rf_model',
    'evaluation',
    'virtual_screening',
    'admet',
    'rdkit_utils',
    'molecular_utils'
]