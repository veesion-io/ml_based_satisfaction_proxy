#!/usr/bin/env python3
"""
Data Exploration Script for Large Parquet Files

This script explores the structure of large parquet files without loading
all data into memory at once.
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

def explore_parquet_structure(file_path: str):
    """
    Explore parquet file structure without loading all data.
    
    Args:
        file_path: Path to the parquet file
    """
    print(f"Exploring structure of {file_path}...")
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File {file_path} not found!")
        return
    
    # Get file size
    file_size = Path(file_path).stat().st_size / (1024**3)  # GB
    print(f"File size: {file_size:.2f} GB")
    
    try:
        # Use pyarrow to get metadata without loading data
        parquet_file = pq.ParquetFile(file_path)
        
        print(f"\nParquet metadata:")
        print(f"Number of row groups: {parquet_file.num_row_groups}")
        print(f"Total rows: {parquet_file.metadata.num_rows}")
        
        # Get schema information
        schema = parquet_file.schema_arrow
        print(f"\nColumns ({len(schema)}):")
        for i, field in enumerate(schema):
            print(f"  {i+1}. {field.name} ({field.type})")
        
        # Read just a small sample to understand the data
        print(f"\nReading first 1000 rows as sample...")
        sample_df = pd.read_parquet(file_path, nrows=1000)
        
        print(f"Sample data shape: {sample_df.shape}")
        print(f"\nFirst 3 rows:")
        print(sample_df.head(3))
        
        print(f"\nData types:")
        print(sample_df.dtypes)
        
        # Check for key columns
        key_columns = ['label', 'probability', 'camera_id', 'store', 'camera']
        found_columns = [col for col in key_columns if col in sample_df.columns]
        missing_columns = [col for col in key_columns if col not in sample_df.columns]
        
        print(f"\nKey columns found: {found_columns}")
        if missing_columns:
            print(f"Key columns missing: {missing_columns}")
        
        # Show unique values for label column if it exists
        if 'label' in sample_df.columns:
            print(f"\nUnique labels in sample (first 20):")
            label_counts = sample_df['label'].value_counts()
            print(label_counts.head(20))
        
        # Show probability statistics if column exists
        prob_col = None
        for col in ['probability', 'prob', 'score']:
            if col in sample_df.columns:
                prob_col = col
                break
        
        if prob_col:
            print(f"\n{prob_col} statistics:")
            print(sample_df[prob_col].describe())
        
        # Memory usage
        print(f"\nMemory usage for sample:")
        print(f"Total: {sample_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Estimate full file memory requirements
        estimated_memory = (sample_df.memory_usage(deep=True).sum() * parquet_file.metadata.num_rows / 1000) / (1024**3)
        print(f"Estimated memory for full file: {estimated_memory:.2f} GB")
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        
        # Try reading with pandas directly but limited rows
        try:
            print(f"\nTrying pandas with limited rows...")
            sample_df = pd.read_parquet(file_path, nrows=100)
            print(f"Successfully read {len(sample_df)} rows")
            print(f"Columns: {list(sample_df.columns)}")
            print(sample_df.head())
        except Exception as e2:
            print(f"Also failed with pandas: {e2}")

def main():
    """Main function to explore the data."""
    input_file = "april-no-autotheft-alerts__raw_20250627_080509.parquet"
    explore_parquet_structure(input_file)

if __name__ == "__main__":
    main() 