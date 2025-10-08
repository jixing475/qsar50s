#!/usr/bin/env python3
"""
QSAR model training script for QSAR50S project.

This script trains ML-QSAR models (ANN, Random Forest) for predicting 
antimicrobial activity against Cutibacterium acnes.

Usage:
    python 03_model_training.py --input data/processed/data_fp_PubChem_train.csv --output models/ --model-type RF --tune
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'qsar50s'))

from models.qsar_models import ANNModel, RFModel
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function for QSAR model training.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train QSAR models for virtual screening of 50S ribosomal inhibitors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Train Random Forest model with hyperparameter tuning
    python 03_model_training.py -i data/processed/data_fp_PubChem_train.csv -o models/ -m RF --tune
    
    # Train ANN model without hyperparameter tuning
    python 03_model_training.py -i data/processed/data_fp_MACCS_train.csv -o models/ -m ANN
    
    # Train model with custom test size and target
    python 03_model_training.py -i data/processed/data_fp_PubChem_train.csv -o models/ -m RF --test-size 0.25 --target pMIC
    
    # Compare both models on the same dataset
    python 03_model_training.py -i data/processed/data_fp_PubChem_train.csv -o models/ -m RF --tune
    python 03_model_training.py -i data/processed/data_fp_PubChem_train.csv -o models/ -m ANN --tune
        '''
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input training data file (CSV format with fingerprints and pMIC values)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for saving trained models'
    )
    
    parser.add_argument(
        '--model-type', '-m',
        choices=['ANN', 'RF'],
        default='RF',
        help='Type of model to train (default: RF)'
    )
    
    parser.add_argument(
        '--target', '-t',
        default='pMIC',
        help='Target variable name (default: pMIC)'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='../config/paths.yaml',
        help='Configuration file path (default: ../config/paths.yaml)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size for validation (default: 0.2)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning (recommended for better performance)'
    )
    
    parser.add_argument(
        '--remove-outliers',
        action='store_true',
        help='Remove outliers using PCA-based detection'
    )
    
    parser.add_argument(
        '--remove-correlated',
        action='store_true',
        help='Remove highly correlated features'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save visualization plots'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("Starting QSAR model training")
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Input file: {args.input}")
        logger.info(f"Output directory: {args.output}")
        
        # Check if input file exists
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        # Initialize model
        if args.model_type == 'ANN':
            model = ANNModel(args.config)
            model_name = "Artificial Neural Network"
        else:
            model = RFModel(args.config)
            model_name = "Random Forest"
        
        logger.info(f"Initialized {model_name} model")
        
        # Load data
        X, y = model.load_data(args.input, args.target)
        
        logger.info(f"Loaded data with shape: {X.shape}")
        logger.info(f"Target variable: {args.target}")
        logger.info(f"Target statistics: mean={y.mean():.3f}, std={y.std():.3f}, min={y.min():.3f}, max={y.max():.3f}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=pd.qcut(y, q=10, duplicates='drop')
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Preprocess features
        logger.info("Preprocessing features...")
        X_train_processed = model.preprocess_features(X_train, fit=True)
        X_test_processed = model.preprocess_features(X_test, fit=False)
        
        # Remove correlated features if requested
        if args.remove_correlated:
            logger.info("Removing highly correlated features...")
            X_train_processed = model.remove_correlated_features(X_train_processed)
            X_test_processed = X_test_processed[X_train_processed.columns]
            logger.info(f"Features after correlation removal: {X_train_processed.shape[1]}")
        
        # Detect and remove outliers if requested
        if args.remove_outliers:
            logger.info("Detecting and removing outliers...")
            X_train_clean, y_train_clean, outlier_indices = model.detect_outliers_pca(
                X_train_processed, y_train, visualize=args.save_plots
            )
            
            logger.info(f"Removed {len(outlier_indices)} outliers")
            logger.info(f"Training set size after outlier removal: {len(X_train_clean)}")
        else:
            X_train_clean, y_train_clean = X_train_processed, y_train
            outlier_indices = []
        
        # Build model
        logger.info("Building model...")
        if args.model_type == 'ANN':
            model.build_model(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                alpha=0.001,
                max_iter=1000
            )
        else:
            model.build_model(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2
            )
        
        # Train model
        logger.info("Training model...")
        performance = model.train_model(X_train_clean, y_train_clean, args.tune)
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        y_pred_test = model.predict(X_test_processed)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        test_performance = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        performance.update(test_performance)
        
        # Create visualizations if requested
        if args.save_plots:
            logger.info("Creating visualization plots...")
            model._create_plots(y_train_clean, y_test, y_pred_test, performance, args.output)
        
        # Save model
        os.makedirs(args.output, exist_ok=True)
        model_path = os.path.join(args.output, f'{args.model_type.lower()}_model.pkl')
        model.save_model(model_path)
        
        # Save performance metrics
        performance_df = pd.DataFrame([performance])
        performance_path = os.path.join(args.output, f'{args.model_type.lower()}_performance.csv')
        performance_df.to_csv(performance_path, index=False)
        
        # Print comprehensive summary
        print(f"\n" + "="*60)
        print(f"{model_name.upper()} MODEL TRAINING SUMMARY")
        print("="*60)
        print(f"Input file: {args.input}")
        print(f"Model type: {model_name}")
        print(f"Target variable: {args.target}")
        print(f"Hyperparameter tuning: {'Yes' if args.tune else 'No'}")
        print(f"Outlier removal: {'Yes' if args.remove_outliers else 'No'}")
        print(f"Correlated feature removal: {'Yes' if args.remove_correlated else 'No'}")
        print("-" * 60)
        print(f"DATA SPLITTING:")
        print(f"  Total samples: {len(X)}")
        print(f"  Training set: {len(X_train_clean)} samples")
        print(f"  Test set: {len(X_test)} samples")
        print(f"  Outliers removed: {len(outlier_indices)} samples")
        print(f"  Test size: {args.test_size:.1%}")
        print("-" * 60)
        print(f"FEATURE INFORMATION:")
        print(f"  Original features: {X.shape[1]}")
        print(f"  Features after preprocessing: {X_train_processed.shape[1]}")
        if args.remove_correlated:
            print(f"  Features after correlation removal: {X_train_clean.shape[1]}")
        print("-" * 60)
        print(f"MODEL PERFORMANCE:")
        print(f"  Training R²: {performance['train_r2']:.4f}")
        print(f"  Test R²: {performance['test_r2']:.4f}")
        print(f"  Training RMSE: {performance['train_rmse']:.4f}")
        print(f"  Test RMSE: {performance['test_rmse']:.4f}")
        print(f"  Training MAE: {performance['train_mae']:.4f}")
        print(f"  Test MAE: {performance['test_mae']:.4f}")
        print(f"  CV RMSE: {performance['cv_rmse_mean']:.4f} ± {performance['cv_rmse_std']:.4f}")
        print("-" * 60)
        print(f"OUTPUT FILES:")
        print(f"  Model saved: {model_path}")
        print(f"  Performance metrics: {performance_path}")
        if args.save_plots:
            print(f"  Plots saved: {args.output}/")
        print("="*60)
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()