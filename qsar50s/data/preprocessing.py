"""
Data preprocessing module for QSAR50S package.

This module handles data loading, cleaning, and preprocessing for ML-QSAR modeling.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import yaml
import logging
from typing import Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Class for handling QSAR data preprocessing.
    """
    
    def __init__(self, config_path: str = "../config/paths.yaml"):
        """
        Initialize the DataProcessor with configuration.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.paths = self.config['paths']
        self.model_params = self.config['model_params']
        
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
    
    def load_chembl_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess ChEMBL data.
        
        Args:
            data_path: Path to the ChEMBL data file (CSV)
            
        Returns:
            Processed DataFrame with SMILES and activity data
        """
        logger.info(f"Loading ChEMBL data from {data_path}")
        
        try:
            # Load data
            data = pd.read_csv(data_path)
            logger.info(f"Original data shape: {data.shape}")
            
            # Select relevant columns
            df_filtered = data[['Molecule ChEMBL ID', 'Smiles', 'Standard Type', 
                               'Standard Relation', 'Standard Value', 'Molecular Weight']].copy()
            
            # Rename columns
            df_filtered.columns = ['ID', 'SMILES', 'Standard_Type', 'Standard_Relation', 
                                 'Standard_Value', 'Molecular_Weight']
            
            # Clean data
            df_filtered['Standard_Relation'] = df_filtered['Standard_Relation'].str.replace("'", "")
            df_filtered['Standard_Value'] = pd.to_numeric(df_filtered['Standard_Value'], errors='coerce')
            df_filtered = df_filtered.dropna(subset=['Standard_Value'])
            
            # Filter for MIC values only
            df_filtered = df_filtered[df_filtered['Standard_Type'] == 'MIC']
            
            # Convert SMILES to string
            df_filtered['SMILES'] = df_filtered['SMILES'].astype(str)
            
            # Remove duplicates, keeping the most potent compound for each ID
            df_filtered = df_filtered.sort_values(['ID', 'Standard_Relation', 'Standard_Value'])
            df_filtered = df_filtered.drop_duplicates('ID', keep='first')
            
            logger.info(f"After filtering: {df_filtered.shape}")
            
            return df_filtered
            
        except Exception as e:
            logger.error(f"Error loading ChEMBL data: {e}")
            raise
    
    def calculate_activity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pMIC values and activity labels.
        
        Args:
            df: DataFrame with MIC values
            
        Returns:
            DataFrame with calculated activity metrics
        """
        logger.info("Calculating activity metrics")
        
        # Convert MIC from ug/mL to mol/L, then calculate pMIC
        # MIC_molar = (MIC_in_ug/mL * 10^-3) / MW
        # pMIC = -log10(MIC_molar)
        
        df['MIC_molar'] = (df['Standard_Value'] * 1e-3) / df['Molecular_Weight']
        df['pMIC'] = -np.log10(df['MIC_molar'])
        
        # Define activity based on pMIC threshold
        activity_cutoff = self.model_params['activity_cutoff']
        valid_relations = ['=', '<', '<=']
        
        df['activity'] = df.apply(
            lambda x: 1 if (x['pMIC'] >= activity_cutoff and x['Standard_Relation'] in valid_relations) else 0,
            axis=1
        )
        
        logger.info(f"Active compounds: {df['activity'].sum()}")
        logger.info(f"Inactive compounds: {(1-df['activity']).sum()}")
        
        return df
    
    def create_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create final dataset with required columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Final processed DataFrame
        """
        logger.info("Creating final dataset")
        
        df_final = df[['ID', 'SMILES', 'pMIC', 'activity']].copy()
        df_final = df_final.sort_values('pMIC', ascending=False)
        df_final['SMILES'] = df_final['SMILES'].astype(str)
        
        return df_final
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save processed data to CSV file.
        
        Args:
            df: DataFrame to save
            output_path: Output file path
        """
        logger.info(f"Saving processed data to {output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved successfully. Shape: {df.shape}")
    
    def plot_activity_distribution(self, df: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """
        Plot activity distribution (requires matplotlib).
        
        Args:
            df: DataFrame with activity data
            output_path: Path to save the plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            
            # Plot histogram
            active_mask = df['activity'] == 1
            inactive_mask = df['activity'] == 0
            
            plt.hist(df[active_mask]['pMIC'], bins=20, alpha=0.7, 
                    color='#E46C0A', label='Active', edgecolor='black')
            plt.hist(df[inactive_mask]['pMIC'], bins=20, alpha=0.7, 
                    color='#3A87C8', label='Inactive', edgecolor='black')
            
            plt.xlabel('pMIC value')
            plt.ylabel('Frequency')
            plt.title('Distribution of pMIC values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Activity distribution plot saved to {output_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available. Skipping plot generation.")
    
    def process_chembl_data(self, input_path: str, output_path: str) -> pd.DataFrame:
        """
        Complete pipeline for processing ChEMBL data.
        
        Args:
            input_path: Path to input ChEMBL data file
            output_path: Path to save processed data
            
        Returns:
            Processed DataFrame
        """
        logger.info("Starting ChEMBL data processing pipeline")
        
        # Load data
        df = self.load_chembl_data(input_path)
        
        # Calculate activity metrics
        df = self.calculate_activity_metrics(df)
        
        # Create final dataset
        df_final = self.create_final_dataset(df)
        
        # Save processed data
        self.save_processed_data(df_final, output_path)
        
        # Generate activity distribution plot
        plot_path = output_path.replace('.csv', '_activity_distribution.png')
        self.plot_activity_distribution(df_final, plot_path)
        
        logger.info("ChEMBL data processing completed successfully")
        
        return df_final


def main():
    """
    Main function for data preprocessing script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess ChEMBL data for QSAR modeling')
    parser.add_argument('--input', '-i', required=True, help='Input ChEMBL data file (CSV)')
    parser.add_argument('--output', '-o', required=True, help='Output processed data file (CSV)')
    parser.add_argument('--config', '-c', default='../config/paths.yaml', help='Configuration file path')
    parser.add_argument('--plot', action='store_true', help='Generate activity distribution plot')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = DataProcessor(args.config)
        
        # Process data
        df_processed = processor.process_chembl_data(args.input, args.output)
        
        print(f"Data preprocessing completed successfully!")
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Active compounds: {df_processed['activity'].sum()}")
        print(f"Inactive compounds: {(1-df_processed['activity']).sum()}")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()