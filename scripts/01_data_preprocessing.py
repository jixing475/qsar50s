#!/usr/bin/env python3
"""
Data preprocessing script for QSAR50S project.

This script processes ChEMBL data to create a dataset suitable for ML-QSAR modeling.
It converts MIC values to pMIC, calculates activity labels, and saves the processed data.

Usage:
    python 01_data_preprocessing.py --input data/raw/chembl_data.csv --output data/processed/training_data.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'qsar50s'))

from data.preprocessing import DataProcessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function for data preprocessing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess ChEMBL data for QSAR modeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Process ChEMBL data with default settings
    python 01_data_preprocessing.py -i data/raw/chembl_data.csv -o data/processed/training_data.csv
    
    # Use custom configuration
    python 01_data_preprocessing.py -i data/raw/chembl_data.csv -o data/processed/training_data.csv -c config/custom_config.yaml
    
    # Generate activity distribution plot
    python 01_data_preprocessing.py -i data/raw/chembl_data.csv -o data/processed/training_data.csv --plot
        '''
    )
    
    parser.add_argument(
        '--input', '-i', 
        required=True,
        help='Input ChEMBL data file (CSV format)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output processed data file (CSV format)'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='../config/paths.yaml',
        help='Configuration file path (default: ../config/paths.yaml)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate activity distribution plot'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("Starting ChEMBL data preprocessing")
        logger.info(f"Input file: {args.input}")
        logger.info(f"Output file: {args.output}")
        logger.info(f"Config file: {args.config}")
        
        # Check if input file exists
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        # Initialize processor
        processor = DataProcessor(args.config)
        
        # Process data
        df_processed = processor.process_chembl_data(args.input, args.output)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("DATA PROCESSING SUMMARY")
        print("="*50)
        print(f"Total compounds processed: {len(df_processed)}")
        print(f"Active compounds (pMIC >= 6): {df_processed['activity'].sum()}")
        print(f"Inactive compounds (pMIC < 6): {(1 - df_processed['activity']).sum()}")
        print(f"Active percentage: {df_processed['activity'].mean() * 100:.1f}%")
        print(f"pMIC range: {df_processed['pMIC'].min():.2f} - {df_processed['pMIC'].max():.2f}")
        print(f"Mean pMIC: {df_processed['pMIC'].mean():.2f}")
        print(f"Std pMIC: {df_processed['pMIC'].std():.2f}")
        print("="*50)
        
        logger.info("Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()