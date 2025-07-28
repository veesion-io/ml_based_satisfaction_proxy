#!/usr/bin/env python3
"""
Model architecture for precision-aware training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .adversary import Adversary

class ResidualMLP(nn.Module):
    def __init__(self, dim, dropout_rate=0.1): 
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x): 
        return F.relu(self.fc2(self.dropout(F.relu(self.fc1(x))))) + x

class MAB(nn.Module):
    def __init__(self, dim_V, num_heads, dropout_rate=0.1, ln=True):
        super(MAB, self).__init__()
        self.mha = nn.MultiheadAttention(dim_V, num_heads, dropout=dropout_rate)
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(dim_V, dim_V), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(dim_V, dim_V)
        )
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, Q, K):
        Q_norm, K_norm = self.ln1(Q).permute(1, 0, 2), self.ln1(K).permute(1, 0, 2)
        out, _ = self.mha(Q_norm, K_norm, K_norm)
        out = Q + self.dropout(out.permute(1, 0, 2))
        out = self.ln2(out)
        out = out + self.ffn(out)
        return out

class DeepSetsPrecisionAware(nn.Module):
    """Deep Sets model for predicting TP/FP densities and TP ratio distribution"""
    def __init__(self, phi_dim=128, n_bins=20, num_heads=4, dropout_rate=0.1):
        super(DeepSetsPrecisionAware, self).__init__()
        
        self.phi = nn.Sequential(
            nn.Linear(3, phi_dim),  # Input: max_proba, is_theft, normalized_count
            nn.ReLU(),
            ResidualMLP(phi_dim, dropout_rate),
            ResidualMLP(phi_dim, dropout_rate),
        )
        # Attention pooling layer
        self.pooling = nn.MultiheadAttention(embed_dim=phi_dim, num_heads=num_heads, batch_first=True)
        
        self.rho = nn.Sequential(
            nn.Linear(phi_dim, phi_dim), # Input is now just the aggregated features
            nn.ReLU(),
            ResidualMLP(phi_dim, dropout_rate),
            ResidualMLP(phi_dim, dropout_rate)
        )
        
        # Output heads for mixture model components
        self.mixture_weights_head = nn.Linear(phi_dim, n_bins)
        self.mixture_locations_head = nn.Linear(phi_dim, n_bins)
        self.mixture_scales_head = nn.Linear(phi_dim, n_bins)
        
        # Adversary
        self.adversary = Adversary(phi_dim)

    def forward(self, x, counts):
        """
        Forward pass for the Deep Sets model.
        x: (batch_size, seq_len, input_dim) - Input features
        counts: (batch_size,) - Number of samples for each item in the batch
        """
        # The input x already contains the normalized counts, so we can pass it directly to phi
        phi_out = self.phi(x)
        
        # Create a query vector for attention pooling
        query = torch.mean(phi_out, dim=1, keepdim=True)
        
        # Attention pooling
        agg, _ = self.pooling(query, phi_out, phi_out)
        agg = agg.squeeze(1)
        
        # Final prediction network
        rho_out = self.rho(agg)
        
        # Get mixture parameters
        mixture_weights = torch.softmax(self.mixture_weights_head(rho_out), dim=1)
        mixture_locations = torch.sigmoid(self.mixture_locations_head(rho_out))
        mixture_scales = F.softplus(self.mixture_scales_head(rho_out))
        
        return mixture_weights, mixture_locations, mixture_scales, rho_out 