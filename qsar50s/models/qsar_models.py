"""
QSAR modeling module for QSAR50S package.

This module provides tools for building and evaluating ML-QSAR models including
ANN, Random Forest, and other machine learning algorithms.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import yaml
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QSARModel:
    """
    Base class for QSAR modeling workflows.
    """
    
    def __init__(self, config_path: str = "../config/paths.yaml"):
        """
        Initialize the QSAR model with configuration.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.model_params = self.config['model_params']
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = VarianceThreshold(threshold=self.model_params['variance_threshold'])
        self.selected_features = None
        self.model_performance = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def load_data(self, data_path: str, target_column: str = 'pMIC') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data for modeling.
        
        Args:
            data_path: Path to the CSV file
            target_column: Name of the target variable column
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info(f"Loading data from {data_path}")
        
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Data shape: {data.shape}")
            
            # Separate features and target
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Remove identifier columns
            feature_columns = [col for col in data.columns if col not in ['Name', 'ID', target_column]]
            X = data[feature_columns]
            y = data[target_column]
            
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Target shape: {y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Preprocess features including variance filtering and scaling.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the transformers (use True for training, False for test)
            
        Returns:
            Preprocessed feature DataFrame
        """
        logger.info("Preprocessing features")
        
        # Remove low variance features
        if fit:
            X_filtered = self.feature_selector.fit_transform(X)
            self.selected_features = X.columns[self.feature_selector.get_support()]
        else:
            X_filtered = self.feature_selector.transform(X)
            X_filtered = pd.DataFrame(X_filtered, columns=self.selected_features)
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X_filtered)
        else:
            X_scaled = self.scaler.transform(X_filtered)
        
        X_scaled = pd.DataFrame(X_scaled, columns=self.selected_features)
        
        logger.info(f"Features after preprocessing: {X_scaled.shape}")
        
        return X_scaled
    
    def remove_correlated_features(self, X: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold (default from config)
            
        Returns:
            DataFrame with uncorrelated features
        """
        if threshold is None:
            threshold = self.model_params['correlation_threshold']
        
        logger.info(f"Removing features with correlation > {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Select features to remove
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        logger.info(f"Removing {len(to_drop)} correlated features")
        
        return X.drop(columns=to_drop)
    
    def detect_outliers_pca(self, X: pd.DataFrame, y: pd.Series, n_components: int = 2, 
                           visualize: bool = True) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
        """
        Detect outliers using PCA and remove them.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_components: Number of PCA components
            visualize: Whether to create visualization
            
        Returns:
            Tuple of (cleaned X, cleaned y, outlier indices)
        """
        from sklearn.decomposition import PCA
        
        logger.info("Detecting outliers using PCA")
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Calculate Mahalanobis distance
        from scipy.spatial.distance import mahalanobis
        from scipy.linalg import inv
        
        cov_matrix = np.cov(X_pca.T)
        inv_cov_matrix = inv(cov_matrix)
        mean = np.mean(X_pca, axis=0)
        
        distances = []
        for point in X_pca:
            distance = mahalanobis(point, mean, inv_cov_matrix)
            distances.append(distance)
        
        # Identify outliers (using 3 standard deviations)
        threshold = np.mean(distances) + 3 * np.std(distances)
        outlier_indices = [i for i, d in enumerate(distances) if d > threshold]
        
        logger.info(f"Detected {len(outlier_indices)} outliers")
        
        # Remove outliers
        X_clean = X.drop(index=outlier_indices)
        y_clean = y.drop(index=outlier_indices)
        
        # Create visualization
        if visualize:
            self._plot_pca_outliers(X_pca, outlier_indices, X_clean.index)
        
        return X_clean, y_clean, outlier_indices
    
    def _plot_pca_outliers(self, X_pca: np.ndarray, outlier_indices: List[int], clean_indices: List[int]) -> None:
        """
        Create PCA plot with outliers highlighted.
        
        Args:
            X_pca: PCA transformed data
            outlier_indices: Indices of outliers
            clean_indices: Indices of clean data points
        """
        plt.figure(figsize=(10, 8))
        
        # Plot clean points
        clean_mask = np.arange(len(X_pca)).isin(clean_indices)
        plt.scatter(X_pca[clean_mask, 0], X_pca[clean_mask, 1], 
                   c='blue', label='Normal', alpha=0.6)
        
        # Plot outliers
        outlier_mask = np.arange(len(X_pca)).isin(outlier_indices)
        plt.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1], 
                   c='red', label='Outliers', alpha=0.8, s=100)
        
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('PCA Outlier Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('../data/processed/pca_outliers.png', dpi=300, bbox_inches='tight')
        plt.show()


class ANNModel(QSARModel):
    """
    Artificial Neural Network model for QSAR.
    """
    
    def __init__(self, config_path: str = "../config/paths.yaml"):
        """
        Initialize the ANN model.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        super().__init__(config_path)
        self.model_name = "ANN"
        
    def build_model(self, hidden_layer_sizes: Tuple[int] = (100, 50), 
                   activation: str = 'relu', solver: str = 'adam',
                   alpha: float = 0.0001, learning_rate_init: float = 0.001,
                   max_iter: int = 500, random_state: int = 42) -> MLPRegressor:
        """
        Build and configure the ANN model.
        
        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes
            activation: Activation function
            solver: Solver for weight optimization
            alpha: L2 penalty parameter
            learning_rate_init: Initial learning rate
            max_iter: Maximum number of iterations
            random_state: Random state for reproducibility
            
        Returns:
            Configured MLPRegressor model
        """
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        )
        
        logger.info(f"Built ANN model with hidden layers: {hidden_layer_sizes}")
        return self.model
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train the ANN model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with model performance metrics
        """
        logger.info("Training ANN model")
        
        if hyperparameter_tuning:
            # Define parameter grid
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
            
            # Perform grid search
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            # Use best model
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            # Cross-validation results
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            
        else:
            # Train with default parameters
            self.model.fit(X_train, y_train)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
        
        # Calculate performance metrics
        y_pred_train = self.model.predict(X_train)
        
        performance = {
            'model_type': 'ANN',
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_r2': r2_score(y_train, y_pred_train),
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std()
        }
        
        self.model_performance = performance
        
        logger.info(f"ANN model trained successfully")
        logger.info(f"Training R²: {performance['train_r2']:.4f}")
        logger.info(f"CV RMSE: {performance['cv_rmse_mean']:.4f} ± {performance['cv_rmse_std']:.4f}")
        
        return performance
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        return self.model.predict(X)
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'performance': self.model_performance,
            'model_name': self.model_name
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.selected_features = model_data['selected_features']
        self.model_performance = model_data['performance']
        self.model_name = model_data['model_name']
        
        logger.info(f"Model loaded from {model_path}")


class RFModel(QSARModel):
    """
    Random Forest model for QSAR.
    """
    
    def __init__(self, config_path: str = "../config/paths.yaml"):
        """
        Initialize the Random Forest model.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        super().__init__(config_path)
        self.model_name = "RandomForest"
        
    def build_model(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                   min_samples_split: int = 2, min_samples_leaf: int = 1,
                   random_state: int = 42) -> RandomForestRegressor:
        """
        Build and configure the Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum number of samples required to split
            min_samples_leaf: Minimum number of samples required at leaf node
            random_state: Random state for reproducibility
            
        Returns:
            Configured RandomForestRegressor model
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        logger.info(f"Built Random Forest model with {n_estimators} trees")
        return self.model
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train the Random Forest model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with model performance metrics
        """
        logger.info("Training Random Forest model")
        
        if hyperparameter_tuning:
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Perform grid search
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            # Use best model
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            # Cross-validation results
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            
        else:
            # Train with default parameters
            self.model.fit(X_train, y_train)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
        
        # Calculate performance metrics
        y_pred_train = self.model.predict(X_train)
        
        performance = {
            'model_type': 'RandomForest',
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_r2': r2_score(y_train, y_pred_train),
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std()
        }
        
        self.model_performance = performance
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            performance['feature_importance'] = feature_importance
            logger.info("Top 5 important features:")
            logger.info(feature_importance.head())
        
        logger.info(f"Random Forest model trained successfully")
        logger.info(f"Training R²: {performance['train_r2']:.4f}")
        logger.info(f"CV RMSE: {performance['cv_rmse_mean']:.4f} ± {performance['cv_rmse_std']:.4f}")
        
        return performance
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        return self.model.predict(X)
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'performance': self.model_performance,
            'model_name': self.model_name
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.selected_features = model_data['selected_features']
        self.model_performance = model_data['performance']
        self.model_name = model_data['model_name']
        
        logger.info(f"Model loaded from {model_path}")


def main():
    """
    Main function for QSAR modeling script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train QSAR models for virtual screening')
    parser.add_argument('--input', '-i', required=True, help='Input data file (CSV)')
    parser.add_argument('--output', '-o', required=True, help='Output directory for models')
    parser.add_argument('--model-type', '-m', choices=['ANN', 'RF'], default='RF', 
                       help='Type of model to train')
    parser.add_argument('--target', '-t', default='pMIC', help='Target variable name')
    parser.add_argument('--config', '-c', default='../config/paths.yaml', help='Configuration file path')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting {args.model_type} model training")
        
        # Initialize model
        if args.model_type == 'ANN':
            model = ANNModel(args.config)
        else:
            model = RFModel(args.config)
        
        # Load data
        X, y = model.load_data(args.input, args.target)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Preprocess features
        X_train_processed = model.preprocess_features(X_train, fit=True)
        X_test_processed = model.preprocess_features(X_test, fit=False)
        
        # Remove correlated features
        X_train_processed = model.remove_correlated_features(X_train_processed)
        X_test_processed = X_test_processed[X_train_processed.columns]
        
        # Detect and remove outliers
        X_train_clean, y_train_clean, outlier_indices = model.detect_outliers_pca(
            X_train_processed, y_train
        )
        
        logger.info(f"Training set size after outlier removal: {len(X_train_clean)}")
        
        # Build model
        if args.model_type == 'ANN':
            model.build_model()
        else:
            model.build_model()
        
        # Train model
        performance = model.train_model(X_train_clean, y_train_clean, args.tune)
        
        # Evaluate on test set
        y_pred_test = model.predict(X_test_processed)
        test_performance = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        performance.update(test_performance)
        
        # Save model
        os.makedirs(args.output, exist_ok=True)
        model_path = os.path.join(args.output, f'{args.model_type.lower()}_model.pkl')
        model.save_model(model_path)
        
        # Print summary
        print(f"\n" + "="*50)
        print(f"{args.model_type} MODEL TRAINING SUMMARY")
        print("="*50)
        print(f"Training set size: {len(X_train_clean)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Outliers removed: {len(outlier_indices)}")
        print(f"Training R²: {performance['train_r2']:.4f}")
        print(f"Test R²: {performance['test_r2']:.4f}")
        print(f"Training RMSE: {performance['train_rmse']:.4f}")
        print(f"Test RMSE: {performance['test_rmse']:.4f}")
        print(f"CV RMSE: {performance['cv_rmse_mean']:.4f} ± {performance['cv_rmse_std']:.4f}")
        print(f"Model saved to: {model_path}")
        print("="*50)
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()