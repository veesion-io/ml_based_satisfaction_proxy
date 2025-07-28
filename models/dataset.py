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
import multiprocessing
from joblib import Parallel, delayed
from torch.nn.utils.rnn import pad_sequence

N_BINS = 20
SAMPLE_SIZE_RANGE = (10, 2000)

class CameraDensityDatasetPrecisionAware(Dataset):
    def __init__(self, processed_data: List[Dict], sample_size_range: Tuple[int, int]):
        self.sample_size_range = sample_size_range
        self.precomputed_data = self._precompute_data(processed_data)

    def _precompute_data(self, data):
        """Precompute and cache data to reduce I/O and computation in __getitem__"""
        print("Caching DataFrames and calculating TP ratios + GT precision...")
        
        # Use joblib for parallel processing
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(
            delayed(self._process_camera_data)(cam_data) for cam_data in tqdm(data, desc="Processing cameras")
        )
        return [res for res in results if res is not None]

    def _process_camera_data(self, cam_data):
        """Helper to process a single camera's data"""
        try:
            df = pd.read_parquet(cam_data['file_path'])
            if len(df) < self.sample_size_range[0]:
                return None
                
            tp_count = df['is_theft'].sum()
            fp_count = len(df) - tp_count
            
            if tp_count == 0 or fp_count == 0:
                return None
            
            gt_precision = tp_count / (tp_count + fp_count)
            
            return {
                'df': df,
                'gt_precision': gt_precision,
                'camera_size': len(df)
            }
        except Exception:
            return None

    def __len__(self):
        """Return total number of cameras"""
        return len(self.precomputed_data)

    def _get_sample_size(self, camera_size):
        """Get a random sample size within the valid range"""
        min_size, max_size = self.sample_size_range
        k = random.randint(min_size, min(max_size, camera_size))
        # Ensure k is not larger than the population size for sampling without replacement
        return min(k, camera_size)

    def _prepare_features(self, sample_df: pd.DataFrame, sample_size: int) -> torch.Tensor:
        """Prepare features"""
        features = sample_df[['max_proba', 'is_theft']].values
        counts_normalized = np.log(sample_size) / np.log(2000)
        counts_expanded = np.full((features.shape[0], 1), counts_normalized)
        return torch.tensor(np.concatenate([features, counts_expanded], axis=1), dtype=torch.float32)

    def __getitem__(self, idx):
        """Return a single data sample"""
        camera = self.precomputed_data[idx]
        df = camera['df']
        camera_size = camera['camera_size']
        
        # Get a random sample size and sample the data
        k = self._get_sample_size(camera_size)
        sample_df = df.sample(n=k, replace=False) # Use replace=False for more realistic sampling
        sample_features = self._prepare_features(sample_df, k)
        
        gt_precision = torch.tensor(camera['gt_precision'], dtype=torch.float32)
        
        return sample_features, gt_precision, torch.tensor(k, dtype=torch.float32)

def collate_fn(batch):
    """Custom collate function for batching variable-length sequences"""
    features, precisions, counts = zip(*[b for b in batch if b is not None])
    if not features: 
        return None, None, None

    # Pad features
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    # Reshape to ensure 3 features
    if padded_features.dim() == 3 and padded_features.size(2) > 3:
        padded_features = padded_features[:, :, :3]

    return padded_features, torch.stack(precisions), torch.tensor(counts, dtype=torch.float32) 