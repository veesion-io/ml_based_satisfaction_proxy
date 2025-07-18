#!/usr/bin/env python3
"""
Data Preprocessing Script for ML-Based Satisfaction Proxy Project

This script processes raw theft alert data from a source parquet file,
creates unified binary labels for theft events, and structures the output
for efficient per-camera analysis in subsequent modeling stages.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define theft labels and their BSR ratios as specified in the plan
LABELS_BSR_RATIOS = {
    "backpack": 0.0073773310368468945,
    "burst shot": 0.0023898990360455025,
    "consumption": 0.005768299027567761,
    "deblistering": 0.0038569216328267135,
    "gesture into body": 0.007369356383484813,
    "other suspicious": 0.002601605299705335,
    "personal bag": 0.008340375083555097,
    "product into stroller": 0.008280012187690432,
    "suspicious bag": 0.0033948719358686615,
    "theft": 0.050290877524495584,
    "suspicious": 0.003580911249770547,
}

THEFT_LABELS = list(LABELS_BSR_RATIOS.keys())

def load_and_process_data(input_file: str, output_file: str = "processed_theft_data.parquet"):
    """
    Load raw parquet data, create binary theft labels, and save processed data.
    
    Args:
        input_file: Path to the input parquet file
        output_file: Path for the output processed parquet file
    
    Returns:
        pandas.DataFrame: Processed DataFrame with binary theft labels
    """
    print(f"Loading data from {input_file}...")
    
    # Load the source parquet file
    df = pd.read_parquet(input_file)
    
    print(f"Loaded {len(df)} rows from source file")
    print(f"Columns in source data: {list(df.columns)}")
    
    # Display basic info about the data
    print(f"\nData info:")
    print(df.info())
    
    # Show first few rows to understand the structure
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Check unique labels in the data
    if 'label' in df.columns:
        unique_labels = df['label'].value_counts()
        print(f"\nUnique labels and their counts:")
        print(unique_labels)
        
        # Show which labels are considered theft vs non-theft
        theft_labels_in_data = [label for label in unique_labels.index if label in THEFT_LABELS]
        non_theft_labels_in_data = [label for label in unique_labels.index if label not in THEFT_LABELS]
        
        print(f"\nTheft labels found in data: {theft_labels_in_data}")
        print(f"Non-theft labels found in data: {non_theft_labels_in_data}")
        
        # Create binary theft label
        print(f"\nCreating binary theft labels...")
        df['is_theft'] = df['label'].isin(THEFT_LABELS).astype(int)
        
        # Show distribution of theft vs non-theft
        theft_distribution = df['is_theft'].value_counts()
        print(f"Theft distribution:")
        print(f"Non-theft (0): {theft_distribution.get(0, 0)}")
        print(f"Theft (1): {theft_distribution.get(1, 0)}")
        print(f"Theft percentage: {(theft_distribution.get(1, 0) / len(df)) * 100:.2f}%")
        
    else:
        print("Warning: 'label' column not found in data!")
        return None
    
    # Identify camera identification columns
    camera_columns = []
    for col in ['camera_id', 'store', 'camera']:
        if col in df.columns:
            camera_columns.append(col)
    
    print(f"\nCamera identification columns found: {camera_columns}")
    
    # Check for probability column
    probability_col = None
    for col in ['probability', 'prob', 'score']:
        if col in df.columns:
            probability_col = col
            break
    
    if probability_col:
        print(f"Probability column found: {probability_col}")
    else:
        print("Warning: No probability column found!")
        return None
    
    # Select relevant columns for the next stage
    essential_columns = camera_columns + [probability_col, 'is_theft']
    
    print(f"\nSelecting essential columns: {essential_columns}")
    processed_df = df[essential_columns].copy()
    
    # Remove any rows with missing values in essential columns
    initial_count = len(processed_df)
    processed_df = processed_df.dropna()
    final_count = len(processed_df)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} rows with missing values")
    
    # Show statistics about cameras
    if camera_columns:
        if len(camera_columns) == 1:
            camera_stats = processed_df[camera_columns[0]].value_counts()
        else:
            # Create a combined camera identifier if multiple columns
            processed_df['camera_id_combined'] = processed_df[camera_columns].astype(str).agg('_'.join, axis=1)
            camera_stats = processed_df['camera_id_combined'].value_counts()
        
        print(f"\nCamera statistics:")
        print(f"Total unique cameras: {len(camera_stats)}")
        print(f"Min alerts per camera: {camera_stats.min()}")
        print(f"Max alerts per camera: {camera_stats.max()}")
        print(f"Mean alerts per camera: {camera_stats.mean():.2f}")
        print(f"Median alerts per camera: {camera_stats.median():.2f}")
    
    # Save processed data
    print(f"\nSaving processed data to {output_file}...")
    processed_df.to_parquet(output_file, index=False)
    
    print(f"Data processing complete!")
    print(f"Final processed data shape: {processed_df.shape}")
    
    return processed_df

def main():
    """Main function to run the data preprocessing."""
    # Define input and output files
    input_file = "april-no-autotheft-alerts__raw_20250627_080509.parquet"
    output_file = "processed_theft_data.parquet"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found!")
        return
    
    # Process the data
    processed_data = load_and_process_data(input_file, output_file)
    
    if processed_data is not None:
        print(f"\n{'='*50}")
        print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
        print(f"{'='*50}")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Processed {len(processed_data)} alert records")

if __name__ == "__main__":
    main() 