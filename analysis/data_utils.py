#!/usr/bin/env python3
"""
Utilities for loading and filtering camera data.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

def load_and_filter_camera_data(data_dir: str = "data_by_camera") -> List[Dict]:
    """
    Loads all camera data, applies selection rules, and returns a list of valid cameras.
    """
    print(f"Searching for camera files in {data_dir}...")
    all_camera_files = [str(f) for f in Path(data_dir).glob("*.parquet")]
    print(f"Found {len(all_camera_files)} camera files.")

    valid_cameras = []
    for file_path in tqdm(all_camera_files, desc="Applying selection rules"):
        df = pd.read_parquet(file_path)
        
        tp_count = df['is_theft'].sum()
        fp_count = len(df) - tp_count

        if len(df) >= 300 and tp_count >= 5 and fp_count >= 5:
            camera_info = {
                'file_path': file_path,
                'tp_count': tp_count,
                'fp_count': fp_count,
                'df': df  # Keep DataFrame in memory for now
            }
            valid_cameras.append(camera_info)
    
    print(f"Found {len(valid_cameras)} cameras that meet the selection criteria.")
    return valid_cameras 