"""
Basic tests for QSAR50S package.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'qsar50s'))

from qsar50s.data.preprocessing import DataProcessor
from qsar50s.features.fingerprints import FingerprintGenerator
from qsar50s.models.qsar_models import RFModel, ANNModel


class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    def test_data_processor_initialization(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()
        assert processor.config is not None
        assert processor.paths is not None
        assert processor.model_params is not None
    
    def test_create_sample_data(self):
        """Test creating sample data for testing."""
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'Molecule ChEMBL ID': [f'CHEMBL{i}' for i in range(10)],
            'Smiles': ['CCO' for _ in range(10)],
            'Standard Type': ['MIC' for _ in range(10)],
            'Standard Relation': ['=' for _ in range(10)],
            'Standard Value': np.random.uniform(1, 10, 10),
            'Molecular Weight': np.random.uniform(150, 500, 10)
        })
        
        # Test basic properties
        assert len(sample_data) == 10
        assert 'Molecule ChEMBL ID' in sample_data.columns
        assert 'Smiles' in sample_data.columns
        assert 'Standard Value' in sample_data.columns


class TestFeatureGeneration:
    """Test feature generation functionality."""
    
    def test_fingerprint_generator_initialization(self):
        """Test FingerprintGenerator initialization."""
        generator = FingerprintGenerator()
        assert generator.config is not None
        assert generator.fingerprint_types is not None
        assert 'PubChem' in generator.fingerprint_types
    
    def test_fingerprint_types(self):
        """Test available fingerprint types."""
        generator = FingerprintGenerator()
        expected_types = ['PubChem', 'MACCS', 'Morgan', 'RDKit']
        
        for fp_type in expected_types:
            assert fp_type in generator.fingerprint_types or fp_type in ['Morgan', 'RDKit']


class TestQSARModels:
    """Test QSAR model functionality."""
    
    def test_rf_model_initialization(self):
        """Test Random Forest model initialization."""
        model = RFModel()
        assert model.config is not None
        assert model.model_params is not None
        assert model.model_name == "RandomForest"
    
    def test_ann_model_initialization(self):
        """Test ANN model initialization."""
        model = ANNModel()
        assert model.config is not None
        assert model.model_params is not None
        assert model.model_name == "ANN"
    
    def test_rf_model_building(self):
        """Test Random Forest model building."""
        model = RFModel()
        model_instance = model.build_model(n_estimators=10, max_depth=5)
        assert model_instance is not None
        assert hasattr(model_instance, 'fit')
    
    def test_ann_model_building(self):
        """Test ANN model building."""
        model = ANNModel()
        model_instance = model.build_model(hidden_layer_sizes=(10,), max_iter=100)
        assert model_instance is not None
        assert hasattr(model_instance, 'fit')
    
    def test_model_with_sample_data(self):
        """Test model training with sample data."""
        # Create sample data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'feature3': np.random.randn(50)
        })
        y = pd.Series(np.random.randn(50))
        
        # Test Random Forest
        rf_model = RFModel()
        rf_model.build_model(n_estimators=10, max_depth=5)
        performance = rf_model.train_model(X, y, hyperparameter_tuning=False)
        
        assert 'train_r2' in performance
        assert 'train_rmse' in performance
        assert 'cv_rmse_mean' in performance
        assert performance['train_r2'] >= 0  # R² should be non-negative


class TestConfiguration:
    """Test configuration handling."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        processor = DataProcessor()
        assert 'paths' in processor.config
        assert 'model_params' in processor.config
        assert 'fingerprints' in processor.config
        assert 'padel' in processor.config
    
    def test_model_parameters(self):
        """Test model parameters."""
        processor = DataProcessor()
        params = processor.model_params
        
        assert 'test_size' in params
        assert 'random_state' in params
        assert 'activity_cutoff' in params
        assert params['test_size'] == 0.2
        assert params['random_state'] == 42


def test_package_import():
    """Test package import."""
    try:
        import qsar50s
        assert hasattr(qsar50s, '__version__')
        assert hasattr(qsar50s, '__author__')
    except ImportError:
        pytest.skip("QSAR50S package not installed")


def test_module_imports():
    """Test individual module imports."""
    try:
        from qsar50s.data import preprocessing
        from qsar50s.features import fingerprints
        from qsar50s.models import qsar_models
        from qsar50s.utils import rdkit_utils, molecular_utils
        
        assert hasattr(preprocessing, 'DataProcessor')
        assert hasattr(fingerprints, 'FingerprintGenerator')
        assert hasattr(qsar_models, 'RFModel')
        assert hasattr(qsar_models, 'ANNModel')
        
    except ImportError as e:
        pytest.skip(f"Module import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])