#!/usr/bin/env python3
"""
Chunked Data Preprocessing Script for Large Parquet Files

This script processes large parquet files in chunks to avoid memory issues.
"""

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

# Define theft labels in French as found in the data
# Based on the sample, these appear to be theft-related labels
FRENCH_THEFT_LABELS = {
    "dissimulation sac",           # bag concealment
    "dissimulation haut du corps", # upper body concealment  
    "dissimulation bas du corps",  # lower body concealment
    "dissimulation landeau",       # stroller concealment
    "dissimulation vetement",      # clothing concealment
    "consommation sur place",      # on-site consumption
    "prise en rafale",            # burst shot
    "geste bizarre",              # bizarre gesture
    "vol",                        # theft
    "sac personnel",              # personal bag
    "sac suspect",                # suspicious bag
    "autre suspect",              # other suspicious
}

# Original English labels from plan (keeping for completeness)
ENGLISH_THEFT_LABELS = {
    "backpack",
    "burst shot", 
    "consumption",
    "deblistering",
    "gesture into body",
    "other suspicious",
    "personal bag",
    "product into stroller",
    "suspicious bag",
    "theft",
    "suspicious",
}

# Combine all theft labels
THEFT_LABELS = FRENCH_THEFT_LABELS | ENGLISH_THEFT_LABELS

def explore_sample_data(file_path: str, num_row_groups: int = 1):
    """
    Read a small sample of the data to understand its structure.
    
    Args:
        file_path: Path to the parquet file
        num_row_groups: Number of row groups to read for sample
    
    Returns:
        pandas.DataFrame: Sample data
    """
    print(f"Reading sample data from {file_path}...")
    
    parquet_file = pq.ParquetFile(file_path)
    
    # Read first row group(s) as sample
    sample_data = []
    for i in range(min(num_row_groups, parquet_file.num_row_groups)):
        row_group = parquet_file.read_row_group(i)
        sample_data.append(row_group.to_pandas())
    
    sample_df = pd.concat(sample_data, ignore_index=True)
    
    print(f"Sample data shape: {sample_df.shape}")
    print(f"Columns: {list(sample_df.columns)}")
    print(f"\nFirst 3 rows:")
    print(sample_df.head(3))
    
    # Check for label column
    label_col = None
    if 'label_id' in sample_df.columns:
        label_col = 'label_id'
    elif 'max_proba_label' in sample_df.columns:
        label_col = 'max_proba_label'
    elif 'label' in sample_df.columns:
        label_col = 'label'
    
    if label_col:
        print(f"\nUsing '{label_col}' as label column")
        print(f"Unique labels in sample:")
        print(sample_df[label_col].value_counts())
        
        # Check which labels are theft vs non-theft (handle NA values)
        sample_labels = sample_df[label_col].dropna().unique()  # Remove NA values
        theft_labels_found = [label for label in sample_labels if label in THEFT_LABELS]
        non_theft_labels_found = [label for label in sample_labels if label not in THEFT_LABELS]
        
        print(f"\nTheft labels in sample: {theft_labels_found}")
        print(f"Non-theft labels in sample: {non_theft_labels_found}")
        
        # Show current theft labels we're looking for
        print(f"\nTheft labels we're looking for:")
        for label in sorted(THEFT_LABELS):
            print(f"  - {label}")
    
    # Check for probability column
    prob_col = None
    if 'max_proba' in sample_df.columns:
        prob_col = 'max_proba'
    elif 'proba' in sample_df.columns:
        prob_col = 'proba'
    elif 'probability' in sample_df.columns:
        prob_col = 'probability'
    
    if prob_col:
        print(f"\nUsing '{prob_col}' as probability column")
        print(f"Probability statistics:")
        print(sample_df[prob_col].describe())
    
    # Check camera identification
    camera_cols = []
    if 'camera_id' in sample_df.columns:
        camera_cols.append('camera_id')
    if 'store_id' in sample_df.columns:
        camera_cols.append('store_id')
    
    print(f"\nCamera identification columns: {camera_cols}")
    
    return sample_df, label_col, prob_col, camera_cols

def process_parquet_in_chunks(input_file: str, output_file: str, chunk_size: int = 1000000):
    """
    Process large parquet file in chunks to avoid memory issues.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output processed file
        chunk_size: Number of rows to process at a time
    """
    print(f"Processing {input_file} in chunks of {chunk_size} rows...")
    
    # First, explore a sample to understand the structure
    sample_df, label_col, prob_col, camera_cols = explore_sample_data(input_file)
    
    if not label_col:
        print("Error: No suitable label column found!")
        return False
    
    if not prob_col:
        print("Error: No suitable probability column found!")
        return False
    
    if not camera_cols:
        print("Error: No camera identification columns found!")
        return False
    
    # Define the columns we need
    essential_columns = camera_cols + [prob_col, label_col]
    print(f"\nProcessing columns: {essential_columns}")
    
    # Initialize counters
    total_processed = 0
    total_theft = 0
    total_non_theft = 0
    chunk_count = 0
    processed_chunks = []
    
    # Open parquet file for reading
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    
    print(f"Total rows to process: {total_rows:,}")
    
    # Process file in row groups (more efficient than arbitrary chunks)
    for i in range(parquet_file.num_row_groups):
        print(f"Processing row group {i+1}/{parquet_file.num_row_groups}...")
        
        # Read row group
        row_group = parquet_file.read_row_group(i)
        chunk_df = row_group.to_pandas()
        
        # Select only essential columns
        chunk_df = chunk_df[essential_columns].copy()
        
        # Remove rows with missing values first
        initial_count = len(chunk_df)
        chunk_df = chunk_df.dropna()
        final_count = len(chunk_df)
        
        if initial_count != final_count:
            print(f"  Removed {initial_count - final_count} rows with missing values")
        
        # Create binary theft label (now safe from NA values)
        chunk_df['is_theft'] = chunk_df[label_col].isin(THEFT_LABELS).astype(int)
        
        # Remove the original label column to save memory
        chunk_df = chunk_df.drop(columns=[label_col])
        
        # Update counters
        chunk_theft = chunk_df['is_theft'].sum()
        chunk_non_theft = len(chunk_df) - chunk_theft
        
        total_processed += len(chunk_df)
        total_theft += chunk_theft
        total_non_theft += chunk_non_theft
        
        print(f"  Processed: {len(chunk_df):,} rows, Theft: {chunk_theft:,}, Non-theft: {chunk_non_theft:,}")
        
        # Store processed chunk
        processed_chunks.append(chunk_df)
        chunk_count += 1
        
        # Save intermediate results every 10 chunks to avoid memory buildup
        if chunk_count % 10 == 0:
            print(f"  Saving intermediate result after {chunk_count} chunks...")
            intermediate_df = pd.concat(processed_chunks, ignore_index=True)
            
            if chunk_count == 10:
                # First save - create new file
                intermediate_df.to_parquet(output_file, index=False)
            else:
                # Append to existing file
                existing_df = pd.read_parquet(output_file)
                combined_df = pd.concat([existing_df, intermediate_df], ignore_index=True)
                combined_df.to_parquet(output_file, index=False)
            
            # Clear processed chunks to free memory
            processed_chunks = []
            
            print(f"  Intermediate save complete. Total processed so far: {total_processed:,}")
    
    # Save any remaining chunks
    if processed_chunks:
        print(f"Saving final chunks...")
        final_df = pd.concat(processed_chunks, ignore_index=True)
        
        if chunk_count <= 10:
            # No previous saves, create new file
            final_df.to_parquet(output_file, index=False)
        else:
            # Append to existing file
            existing_df = pd.read_parquet(output_file)
            combined_df = pd.concat([existing_df, final_df], ignore_index=True)
            combined_df.to_parquet(output_file, index=False)
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Total rows processed: {total_processed:,}")
    print(f"Theft alerts: {total_theft:,} ({100*total_theft/total_processed:.2f}%)")
    print(f"Non-theft alerts: {total_non_theft:,} ({100*total_non_theft/total_processed:.2f}%)")
    print(f"Output saved to: {output_file}")
    
    # Show final statistics
    final_df = pd.read_parquet(output_file)
    print(f"\nFinal output shape: {final_df.shape}")
    print(f"Final columns: {list(final_df.columns)}")
    
    # Camera statistics
    if len(camera_cols) == 1:
        camera_stats = final_df[camera_cols[0]].value_counts()
    else:
        # Create combined camera ID
        final_df['camera_id_combined'] = final_df[camera_cols].astype(str).agg('_'.join, axis=1)
        camera_stats = final_df['camera_id_combined'].value_counts()
    
    print(f"\nCamera statistics:")
    print(f"Total unique cameras: {len(camera_stats):,}")
    print(f"Min alerts per camera: {camera_stats.min()}")
    print(f"Max alerts per camera: {camera_stats.max()}")
    print(f"Mean alerts per camera: {camera_stats.mean():.2f}")
    print(f"Median alerts per camera: {camera_stats.median():.2f}")
    
    return True

def main():
    """Main function to run the chunked data preprocessing."""
    input_file = "april-no-autotheft-alerts__raw_20250627_080509.parquet"
    output_file = "processed_theft_data.parquet"
    
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found!")
        return
    
    success = process_parquet_in_chunks(input_file, output_file)
    
    if success:
        print(f"\n✅ Data preprocessing completed successfully!")
        print(f"📁 Processed data saved to: {output_file}")
    else:
        print(f"\n❌ Data preprocessing failed!")

if __name__ == "__main__":
    main() 