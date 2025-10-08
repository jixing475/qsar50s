#!/usr/bin/env python3
"""
Feature generation script for QSAR50S project.

This script generates molecular fingerprints and descriptors from SMILES data
using RDKit and PaDEL-Descriptor.

Usage:
    python 02_feature_generation.py --input data/processed/training_data.csv --output data/processed --fingerprint-type PubChem --method padel
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'qsar50s'))

from features.fingerprints import FingerprintGenerator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function for feature generation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate molecular fingerprints and descriptors for QSAR modeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Generate PubChem fingerprints using PaDEL-Descriptor for training data
    python 02_feature_generation.py -i data/processed/training_data.csv -o data/processed -f PubChem -m padel -d train
    
    # Generate MACCS fingerprints using RDKit for test data
    python 02_feature_generation.py -i data/processed/test_data.csv -o data/processed -f MACCS -m rdkit -d test
    
    # Generate fingerprints from SMI file
    python 02_feature_generation.py --smi-input data/external/library.smi --output data/processed -f PubChem
        '''
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Input data file (CSV format with SMILES and ID columns)'
    )
    
    parser.add_argument(
        '--smi-input', '-s',
        help='Input SMI file (SMILES format)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for generated features'
    )
    
    parser.add_argument(
        '--fingerprint-type', '-f',
        default='PubChem',
        choices=['PubChem', 'MACCS', 'Morgan', 'RDKit', 'AtomPairs2D', 'EState', 'CDKextended', 'KlekotaRoth'],
        help='Type of fingerprint to generate (default: PubChem)'
    )
    
    parser.add_argument(
        '--method', '-m',
        default='padel',
        choices=['padel', 'rdkit'],
        help='Method to use for fingerprint generation (default: padel)'
    )
    
    parser.add_argument(
        '--dataset-type', '-d',
        default='train',
        choices=['train', 'test'],
        help='Dataset type (default: train)'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='../config/paths.yaml',
        help='Configuration file path (default: ../config/paths.yaml)'
    )
    
    parser.add_argument(
        '--generate-descriptors',
        action='store_true',
        help='Also generate molecular descriptors using RDKit'
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
        logger.info("Starting molecular feature generation")
        
        # Initialize generator
        generator = FingerprintGenerator(args.config)
        
        # Check if we have valid input
        if not args.input and not args.smi_input:
            raise ValueError("Either --input or --smi-input must be provided")
        
        if args.smi_input:
            # Process SMI file
            logger.info(f"Processing SMI file: {args.smi_input}")
            logger.info(f"Fingerprint type: {args.fingerprint_type}")
            
            if not os.path.exists(args.smi_input):
                raise FileNotFoundError(f"SMI file not found: {args.smi_input}")
            
            # Generate fingerprints from SMI file
            descriptors = generator.generate_fingerprints_from_smi(
                args.smi_input, 
                args.fingerprint_type,
                output_prefix=os.path.join(args.output, 'fingerprints')
            )
            
            print(f"\n" + "="*50)
            print("FINGERPRINT GENERATION SUMMARY")
            print("="*50)
            print(f"Input SMI file: {args.smi_input}")
            print(f"Fingerprint type: {args.fingerprint_type}")
            print(f"Generated features: {descriptors.shape[0]} molecules × {descriptors.shape[1]-1} features")
            print(f"Output directory: {args.output}")
            print("="*50)
            
        else:
            # Process CSV file
            logger.info(f"Processing CSV file: {args.input}")
            logger.info(f"Fingerprint type: {args.fingerprint_type}")
            logger.info(f"Method: {args.method}")
            logger.info(f"Dataset type: {args.dataset_type}")
            
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Input file not found: {args.input}")
            
            # Load data
            import pandas as pd
            df = pd.read_csv(args.input)
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Check required columns
            required_cols = ['SMILES', 'ID']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Generate fingerprints
            if args.method == 'padel':
                descriptors = generator.generate_padel_fingerprints(
                    df, args.fingerprint_type, args.dataset_type
                )
            else:  # rdkit
                descriptors = generator.generate_rdkit_fingerprints(
                    df, args.fingerprint_type
                )
            
            # Save results
            result_data = generator.save_ml_data(
                df, descriptors, args.dataset_type, args.fingerprint_type, args.output
            )
            
            # Generate additional descriptors if requested
            if args.generate_descriptors and args.method == 'rdkit':
                logger.info("Generating additional molecular descriptors")
                descriptor_data = generator.generate_rdkit_descriptors(df)
                descriptor_output = os.path.join(
                    args.output, f'descriptors_{args.fingerprint_type}_{args.dataset_type}.csv'
                )
                descriptor_data.to_csv(descriptor_output, index=False)
                logger.info(f"Additional descriptors saved to: {descriptor_output}")
            
            print(f"\n" + "="*50)
            print("FEATURE GENERATION SUMMARY")
            print("="*50)
            print(f"Input file: {args.input}")
            print(f"Fingerprint type: {args.fingerprint_type}")
            print(f"Method: {args.method}")
            print(f"Dataset type: {args.dataset_type}")
            print(f"Original data shape: {df.shape}")
            print(f"Generated features: {descriptors.shape[0]} molecules × {descriptors.shape[1]-1} features")
            print(f"Output data shape: {result_data.shape}")
            print(f"Output directory: {args.output}")
            
            if args.generate_descriptors and args.method == 'rdkit':
                print(f"Additional descriptors: {descriptor_data.shape[1]-len(required_cols)} molecular descriptors")
            
            print("="*50)
        
        logger.info("Feature generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()