#!/usr/bin/env python3
"""
Dataset classes for precision-aware training
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from tqdm import tqdm
import random

N_BINS = 20
SAMPLE_SIZE_RANGE = (10, 2000)

class CameraDensityDatasetPrecisionAware(Dataset):
    def __init__(self, processed_data: List[Dict], sample_size_range: Tuple[int, int]):
        self.sample_size_range = sample_size_range
        self.camera_data = processed_data
        self.df_cache = {}
        
        # Cache DataFrames and calculate TP ratios + ground truth precision
        print("Caching DataFrames and calculating TP ratios + GT precision...")
        for cam in tqdm(processed_data, desc="Processing cameras"):
            df = pd.read_parquet(cam['file_path'])
            self.df_cache[cam['file_path']] = df
            
            # Calculate TP ratio
            total_alerts = len(df)
            tp_alerts = len(df[df['is_theft'] == 1])
            tp_ratio = tp_alerts / total_alerts if total_alerts > 0 else 0.0
            cam['tp_ratio'] = tp_ratio
            
            # Calculate ground truth TP ratio (simple precision)
            gt_precision = self.calculate_ground_truth_precision(df)
            cam['gt_precision'] = gt_precision

    def calculate_ground_truth_precision(self, camera_data):
        """Calculate ground truth TP ratio for a camera (same as what model should predict)"""
        tp_count = len(camera_data[camera_data['is_theft'] == 1])
        total_count = len(camera_data)
        
        if total_count == 0:
            return 0.0
        
        return tp_count / total_count

    def __len__(self):
        return len(self.camera_data)

    def _get_sample_size(self, camera_size: int) -> int:
        """Calculate sample size using smart sampling strategy"""
        min_size, max_size = self.sample_size_range
        
        if random.random() < 0.7:
            max_pct = min(0.8, max_size / camera_size) if camera_size > 0 else 0.8
            pct = random.uniform(0.01, max_pct)
            k = max(min_size, int(camera_size * pct))
        else:
            k = random.randint(min_size, min(max_size, camera_size))
        
        return min(k, camera_size)

    def _prepare_features(self, sample_df: pd.DataFrame) -> torch.Tensor:
        """Prepare features"""
        return torch.tensor(sample_df[['max_proba', 'is_theft']].values, dtype=torch.float32)

    def __getitem__(self, idx):
        camera = self.camera_data[idx]
        df = self.df_cache[camera['file_path']]
        camera_size = len(df)
        
        # Smart sampling
        k = self._get_sample_size(camera_size)
        sample_df = df.sample(n=k, replace=False)
        sample_features = self._prepare_features(sample_df)
        
        tp_density = torch.tensor(camera['tp_density'], dtype=torch.float32)
        fp_density = torch.tensor(camera['fp_density'], dtype=torch.float32)
        tp_ratio = torch.tensor(camera['tp_ratio'], dtype=torch.float32)
        gt_precision = torch.tensor(camera['gt_precision'], dtype=torch.float32)
        
        return sample_features, tp_density, fp_density, tp_ratio, gt_precision, k

def collate_fn(batch):
    """Custom collate function for batching variable-length sequences"""
    features, tps, fps, ratios, precisions, counts = zip(*[b for b in batch if b is not None])
    if not features: 
        return None, None, None, None, None, None
    
    max_len = max(len(f) for f in features)
    padded = torch.zeros(len(features), max_len, 2)
    for i, f in enumerate(features): 
        padded[i, :len(f), :] = f
    
    return (padded, torch.stack(tps), torch.stack(fps), 
            torch.stack(ratios), torch.stack(precisions), torch.tensor(counts, dtype=torch.float32)) 