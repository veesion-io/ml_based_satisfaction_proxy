#!/usr/bin/env python3
"""
Data Partitioning Script

This script takes the single large processed parquet file and partitions it
into a directory structure where each camera's data is stored in its own
separate, small parquet file.

This allows for highly memory-efficient loading during training, as only
the data for a single camera needs to be in memory at any given time.
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

def partition_data_by_camera(input_file: str, output_dir: str):
    """
    Reads a large parquet file and saves its data into separate files per camera.

    Args:
        input_file: Path to the large processed parquet file.
        output_dir: Directory to save the partitioned files.
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input file not found at {input_file}")
        return

    # Create the output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    print(f"Partitioning data from {input_file} into directory {output_dir}...")

    # First, get all unique camera IDs without loading all data
    print("Finding unique camera IDs...")
    df_cameras = pd.read_parquet(input_file, columns=['camera_id', 'store_id'])
    df_cameras['camera_id_combined'] = df_cameras['camera_id'].astype(str) + '_' + df_cameras['store_id'].astype(str)
    unique_cameras = df_cameras['camera_id_combined'].unique()
    print(f"Found {len(unique_cameras)} unique cameras.")

    # Now, iterate through the large file and save data for each camera
    # This is memory intensive but is a one-time cost. A chunked approach would be better
    # for extreme memory constraints, but let's try this first.
    print("Loading full dataset to partition...")
    df = pd.read_parquet(input_file)
    df['camera_id_combined'] = df['camera_id'].astype(str) + '_' + df['store_id'].astype(str)
    
    # Group by the combined camera ID
    grouped = df.groupby('camera_id_combined')

    # Save each group to its own file
    for camera_id, group_df in tqdm(grouped, desc="Partitioning data by camera"):
        # Sanitize the camera_id to be a valid filename
        safe_camera_id = "".join([c if c.isalnum() else "_" for c in camera_id])
        file_path = output_path / f"{safe_camera_id}.parquet"
        
        # Select only the necessary columns to save space
        final_df = group_df[['max_proba', 'is_theft']].copy()
        
        final_df.to_parquet(file_path, index=False)
        
    print("\n✅ Data partitioning complete.")
    print(f"Data for {len(unique_cameras)} cameras has been saved to '{output_dir}'.")

def main():
    """Main function to run the partitioning."""
    input_file = "processed_theft_data.parquet"
    output_dir = "data_by_camera"
    partition_data_by_camera(input_file, output_dir)

if __name__ == "__main__":
    main() 