"""
Molecular fingerprint and descriptor generation module for QSAR50S package.

This module handles the calculation of molecular fingerprints and descriptors using
various tools including RDKit and PaDEL-Descriptor.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import PandasTools, AllChem, MACCSkeys, Descriptors
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Some functions will be limited.")
    RDKIT_AVAILABLE = False

# Import padelpy
try:
    from padelpy import padeldescriptor, from_smiles
    PADELPY_AVAILABLE = True
except ImportError:
    logger.warning("padelpy not available. PaDEL-Descriptor functions will be limited.")
    PADELPY_AVAILABLE = False


class FingerprintGenerator:
    """
    Class for generating molecular fingerprints using various methods.
    """
    
    def __init__(self, config_path: str = "../config/paths.yaml"):
        """
        Initialize the FingerprintGenerator with configuration.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.paths = self.config['paths']
        self.fingerprint_settings = self.config['fingerprints']
        self.padel_settings = self.config['padel']
        
        # Define available fingerprint types
        self.fingerprint_types = {
            'AtomPairs2DCount': 'AtomPairs2DFingerprintCount.xml',
            'AtomPairs2D': 'AtomPairs2DFingerprinter.xml',
            'EState': 'EStateFingerprinter.xml',
            'CDKextended': 'ExtendedFingerprinter.xml',
            'CDK': 'Fingerprinter.xml',
            'CDKgraphonly': 'GraphOnlyFingerprinter.xml',
            'KlekotaRothCount': 'KlekotaRothFingerprintCount.xml',
            'KlekotaRoth': 'KlekotaRothFingerprinter.xml',
            'MACCS': 'MACCSFingerprinter.xml',
            'PubChem': 'PubchemFingerprinter.xml',
            'SubstructureCount': 'SubstructureFingerprintCount.xml',
            'Substructure': 'SubstructureFingerprinter.xml'
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration
        """
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except ImportError:
            logger.error("PyYAML not available. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'paths': {
                'external_data': '../data/external'
            },
            'fingerprints': {
                'default_type': 'PubChem',
                'radius': 2,
                'n_bits': 1024
            },
            'padel': {
                'threads': 2,
                'remove_salt': True,
                'standardize_nitro': True,
                'standardize_tautomers': True,
                'detect_aromaticity': True
            }
        }
    
    def generate_rdkit_fingerprints(self, df: pd.DataFrame, fingerprint_type: str = 'Morgan') -> pd.DataFrame:
        """
        Generate fingerprints using RDKit.
        
        Args:
            df: DataFrame with SMILES column
            fingerprint_type: Type of fingerprint ('Morgan', 'MACCS', 'RDKit')
            
        Returns:
            DataFrame with fingerprint columns
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit not available for fingerprint generation")
        
        logger.info(f"Generating {fingerprint_type} fingerprints using RDKit")
        
        # Add molecule column
        df_mol = df.copy()
        PandasTools.AddMoleculeColumnToFrame(df_mol, 'SMILES', 'ROMol')
        
        if fingerprint_type == 'Morgan':
            return self._generate_morgan_fingerprints(df_mol)
        elif fingerprint_type == 'MACCS':
            return self._generate_maccs_fingerprints(df_mol)
        elif fingerprint_type == 'RDKit':
            return self._generate_rdkit_fingerprints(df_mol)
        else:
            raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")
    
    def _generate_morgan_fingerprints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Morgan fingerprints."""
        radius = self.fingerprint_settings['radius']
        n_bits = self.fingerprint_settings['n_bits']
        
        def mol_to_morgan_fp(mol):
            try:
                if mol is None:
                    return None
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                return np.array(fp)
            except:
                return None
        
        fp_data = df['ROMol'].apply(mol_to_morgan_fp)
        fp_cols = [f'morgan_{i}' for i in range(n_bits)]
        fp_df = pd.DataFrame(fp_data.tolist(), columns=fp_cols, index=df.index)
        
        return pd.concat([df.drop('ROMol', axis=1), fp_df], axis=1)
    
    def _generate_maccs_fingerprints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate MACCS fingerprints."""
        def mol_to_maccs_fp(mol):
            try:
                if mol is None:
                    return None
                fp = MACCSkeys.GenMACCSKeys(mol)
                return np.array(fp)
            except:
                return None
        
        fp_data = df['ROMol'].apply(mol_to_maccs_fp)
        fp_cols = [f'maccs_{i}' for i in range(167)]  # MACCS has 167 bits
        fp_df = pd.DataFrame(fp_data.tolist(), columns=fp_cols, index=df.index)
        
        return pd.concat([df.drop('ROMol', axis=1), fp_df], axis=1)
    
    def _generate_rdkit_fingerprints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RDKit topological fingerprints."""
        def mol_to_rdkit_fp(mol):
            try:
                if mol is None:
                    return None
                fp = Chem.RDKFingerprint(mol)
                return np.array(fp)
            except:
                return None
        
        fp_data = df['ROMol'].apply(mol_to_rdkit_fp)
        n_bits = self.fingerprint_settings['n_bits']
        fp_cols = [f'rdkit_{i}' for i in range(n_bits)]
        fp_df = pd.DataFrame(fp_data.tolist(), columns=fp_cols, index=df.index)
        
        return pd.concat([df.drop('ROMol', axis=1), fp_df], axis=1)
    
    def generate_padel_fingerprints(self, df: pd.DataFrame, fingerprint_type: str = 'PubChem',
                                  dataset_type: str = 'train') -> pd.DataFrame:
        """
        Generate fingerprints using PaDEL-Descriptor.
        
        Args:
            df: DataFrame with SMILES and ID columns
            fingerprint_type: Type of fingerprint from PaDEL
            dataset_type: Type of dataset ('train' or 'test')
            
        Returns:
            DataFrame with fingerprint columns
        """
        if not PADELPY_AVAILABLE:
            raise ImportError("padelpy not available for fingerprint generation")
        
        logger.info(f"Generating {fingerprint_type} fingerprints using PaDEL-Descriptor")
        
        # Check if fingerprint type is valid
        if fingerprint_type not in self.fingerprint_types:
            raise ValueError(f"Invalid fingerprint type: {fingerprint_type}")
        
        # Create temporary SMI file
        smi_file = f'temp_molecule_{dataset_type}.smi'
        df_mol = pd.concat([df['SMILES'], df['ID']], axis=1)
        df_mol.to_csv(smi_file, sep='\t', index=False, header=False)
        
        # Get XML file path
        xml_filename = self.fingerprint_types[fingerprint_type]
        # Case insensitive file search
        xml_file = None
        for file in os.listdir(self.paths['external_data']):
            if file.lower() == xml_filename.lower():
                xml_file = os.path.join(self.paths['external_data'], file)
                break
        
        if xml_file is None or not os.path.exists(xml_file):
            raise FileNotFoundError(f"Fingerprint XML file not found: {xml_file}")
        
        # Generate fingerprints
        output_file = f'{fingerprint_type}_{dataset_type}.csv'
        
        try:
            padeldescriptor(
                mol_dir=smi_file,
                d_file=output_file,
                descriptortypes=xml_file,
                detectaromaticity=self.padel_settings['detect_aromaticity'],
                standardizenitro=self.padel_settings['standardize_nitro'],
                standardizetautomers=self.padel_settings['standardize_tautomers'],
                threads=self.padel_settings['threads'],
                removesalt=self.padel_settings['remove_salt'],
                log=True,
                fingerprints=True
            )
            
            # Read generated fingerprints
            descriptors = pd.read_csv(output_file)
            logger.info(f"Generated {descriptors.shape[0]} molecules with {descriptors.shape[1]-1} descriptors")
            
            # Clean up temporary file
            if os.path.exists(smi_file):
                os.remove(smi_file)
            
            return descriptors
            
        except Exception as e:
            logger.error(f"Error generating PaDEL fingerprints: {e}")
            # Clean up temporary file
            if os.path.exists(smi_file):
                os.remove(smi_file)
            raise
    
    def generate_fingerprints_from_smi(self, smi_file_path: str, fingerprint_type: str = 'PubChem',
                                    output_prefix: str = 'fingerprints') -> pd.DataFrame:
        """
        Generate fingerprints directly from a SMI file using PaDEL-Descriptor.
        
        Args:
            smi_file_path: Path to the SMI file
            fingerprint_type: Type of fingerprint
            output_prefix: Prefix for output files
            
        Returns:
            DataFrame with generated fingerprints
        """
        if not PADELPY_AVAILABLE:
            raise ImportError("padelpy not available for fingerprint generation")
        
        # Check if SMI file exists
        if not os.path.exists(smi_file_path):
            raise FileNotFoundError(f"SMI file not found: {smi_file_path}")
        
        # Check if fingerprint type is valid
        if fingerprint_type not in self.fingerprint_types:
            raise ValueError(f"Invalid fingerprint type: {fingerprint_type}")
        
        logger.info(f"Processing SMI file: {smi_file_path}")
        logger.info(f"Fingerprint type: {fingerprint_type}")
        
        # Get XML file path
        xml_filename = self.fingerprint_types[fingerprint_type]
        # Case insensitive file search
        xml_file = None
        for file in os.listdir(self.paths['external_data']):
            if file.lower() == xml_filename.lower():
                xml_file = os.path.join(self.paths['external_data'], file)
                break
        
        if xml_file is None or not os.path.exists(xml_file):
            raise FileNotFoundError(f"Fingerprint XML file not found: {xml_file}")
        
        # Generate fingerprints
        output_file = f'{output_prefix}_{fingerprint_type}.csv'
        
        try:
            padeldescriptor(
                mol_dir=smi_file_path,
                d_file=output_file,
                descriptortypes=xml_file,
                detectaromaticity=self.padel_settings['detect_aromaticity'],
                standardizenitro=self.padel_settings['standardize_nitro'],
                standardizetautomers=self.padel_settings['standardize_tautomers'],
                threads=self.padel_settings['threads'],
                removesalt=self.padel_settings['remove_salt'],
                log=True,
                fingerprints=True
            )
            
            # Read generated fingerprints
            descriptors = pd.read_csv(output_file)
            logger.info(f"Generated fingerprints for {descriptors.shape[0]} molecules with {descriptors.shape[1]-1} descriptors")
            logger.info(f"Output saved to: {output_file}")
            
            return descriptors
            
        except Exception as e:
            logger.error(f"Error generating fingerprints from SMI file: {e}")
            raise
    
    def save_ml_data(self, df_original: pd.DataFrame, descriptors: pd.DataFrame,
                    dataset_type: str, fingerprint_type: str = 'PubChem',
                    output_dir: str = "../data/processed") -> pd.DataFrame:
        """
        Combine descriptors with original data and save for machine learning.
        
        Args:
            df_original: Original DataFrame with ID and target variable
            descriptors: DataFrame with descriptors from PaDEL
            dataset_type: Type of dataset ('train' or 'test')
            fingerprint_type: Type of fingerprint used
            output_dir: Output directory
            
        Returns:
            Combined DataFrame
        """
        id_col = df_original['ID'].rename('Name')
        
        # Handle both PaDEL and RDKit outputs
        if 'Name' in descriptors.columns:
            # PaDEL output format
            X = descriptors.drop('Name', axis=1)
        else:
            # RDKit output format - might have SMILES and duplicate pMIC
            X = descriptors.copy()
            # Remove SMILES column if present
            if 'SMILES' in X.columns:
                X = X.drop('SMILES', axis=1)
            # Remove duplicate pMIC column if present
            if 'pMIC' in X.columns and dataset_type == 'train':
                X = X.drop('pMIC', axis=1)
        
        # For train set, include target variable; for test set, only features
        if dataset_type == 'train':
            y = df_original['pMIC']
            data = pd.concat([id_col, X, y], axis=1)
        else:  # test set
            data = pd.concat([id_col, X], axis=1)
        
        # Save with clear naming convention
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f'data_fp_{fingerprint_type}_{dataset_type}.csv')
        data.to_csv(output_filename, index=False)
        
        logger.info(f"Data saved to {output_filename}")
        return data
    
    def generate_rdkit_descriptors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate molecular descriptors using RDKit.
        
        Args:
            df: DataFrame with SMILES column
            
        Returns:
            DataFrame with descriptor columns
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit not available for descriptor generation")
        
        logger.info("Generating RDKit molecular descriptors")
        
        # Add molecule column
        df_mol = df.copy()
        PandasTools.AddMoleculeColumnToFrame(df_mol, 'SMILES', 'ROMol')
        
        # Calculate descriptors
        descriptors_list = []
        for mol in df_mol['ROMol']:
            if mol is not None:
                try:
                    descriptors = {
                        'MolWt': Descriptors.MolWt(mol),
                        'LogP': Descriptors.MolLogP(mol),
                        'TPSA': Descriptors.TPSA(mol),
                        'NumHDonors': Descriptors.NumHDonors(mol),
                        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                        'NumRings': rdMolDescriptors.CalcNumRings(mol),
                        'MolMR': Descriptors.MolMR(mol),
                        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
                        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
                        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),
                        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
                        'NumHBD': Descriptors.NumHDonors(mol),
                        'NumHBA': Descriptors.NumHAcceptors(mol)
                    }
                except:
                    descriptors = {key: np.nan for key in [
                        'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 
                        'NumRotatableBonds', 'NumRings', 'MolMR', 'NumHeteroatoms',
                        'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
                        'NumHBD', 'NumHBA'
                    ]}
            else:
                descriptors = {key: np.nan for key in [
                    'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 
                    'NumRotatableBonds', 'NumRings', 'MolMR', 'NumHeteroatoms',
                    'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
                    'NumHBD', 'NumHBA'
                ]}
            
            descriptors_list.append(descriptors)
        
        descriptors_df = pd.DataFrame(descriptors_list)
        
        return pd.concat([df_mol.drop('ROMol', axis=1), descriptors_df], axis=1)


def main():
    """
    Main function for fingerprint generation script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate molecular fingerprints and descriptors')
    parser.add_argument('--input', '-i', required=True, help='Input data file (CSV)')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--fingerprint-type', '-f', default='PubChem', 
                       help='Fingerprint type (PubChem, MACCS, Morgan, etc.)')
    parser.add_argument('--method', '-m', default='padel', 
                       choices=['padel', 'rdkit'], help='Method to use for fingerprint generation')
    parser.add_argument('--dataset-type', '-d', default='train',
                       choices=['train', 'test'], help='Dataset type')
    parser.add_argument('--config', '-c', default='../config/paths.yaml', 
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = FingerprintGenerator(args.config)
        
        # Load data
        df = pd.read_csv(args.input)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Generate fingerprints
        if args.method == 'padel':
            descriptors = generator.generate_padel_fingerprints(
                df, args.fingerprint_type, args.dataset_type
            )
        else:  # rdkit
            descriptors = generator.generate_rdkit_fingerprints(df, args.fingerprint_type)
        
        # Save results
        generator.save_ml_data(
            df, descriptors, args.dataset_type, args.fingerprint_type, args.output
        )
        
        logger.info("Fingerprint generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Fingerprint generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()