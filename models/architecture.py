#!/usr/bin/env python3
"""
Model architecture for precision-aware training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, phi_dim, n_bins, num_heads, dropout_rate=0.1):
        super(DeepSetsPrecisionAware, self).__init__()
        if phi_dim % num_heads != 0: 
            phi_dim = (phi_dim // num_heads + 1) * num_heads
        
        self.phi = nn.Sequential(
            nn.Linear(2, phi_dim),
            nn.Dropout(dropout_rate),
            ResidualMLP(phi_dim, dropout_rate), 
            ResidualMLP(phi_dim, dropout_rate)
        )
        self.pooling = MAB(phi_dim, num_heads, dropout_rate)
        self.query = nn.Parameter(torch.randn(1, 1, phi_dim))
        self.rho = nn.Sequential(
            ResidualMLP(phi_dim, dropout_rate), 
            ResidualMLP(phi_dim, dropout_rate)
        )
        
        # Initialize output heads
        self._init_output_heads(phi_dim, n_bins, dropout_rate)
    
    def _init_output_heads(self, phi_dim, n_bins, dropout_rate):
        """Initialize the three output heads"""
        # TP and FP density heads
        self.tp_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(phi_dim, n_bins)
        )
        self.fp_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(phi_dim, n_bins)
        )
        
        # Mixture of logistic components for TP ratio distribution
        self.num_mixture_components = 5
        self.mixture_weights_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(phi_dim, self.num_mixture_components)
        )
        self.mixture_locations_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(phi_dim, self.num_mixture_components)
        )
        self.mixture_scales_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(phi_dim, self.num_mixture_components)
        )
    
    def forward(self, x, counts):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < counts[:, None]
        phi_out = self.phi(x * mask.unsqueeze(-1))
        query = self.query.repeat(x.shape[0], 1, 1)
        agg = self.pooling(query, phi_out).squeeze(1)
        rho_out = self.rho(agg)
        
        tp_out = F.softmax(self.tp_head(rho_out), dim=1)
        fp_out = F.softmax(self.fp_head(rho_out), dim=1)
        
        # Mixture of logistic components for TP ratio distribution
        mixture_weights_raw = self.mixture_weights_head(rho_out)
        mixture_locations_raw = self.mixture_locations_head(rho_out)
        mixture_scales_raw = self.mixture_scales_head(rho_out)
        
        # Normalize mixture weights
        mixture_weights = F.softmax(mixture_weights_raw, dim=1)
        
        # Constrain locations to [0, 1] using sigmoid
        mixture_locations = torch.sigmoid(mixture_locations_raw)
        
        # Constrain scales to be positive using softplus
        mixture_scales = F.softplus(mixture_scales_raw) + 1e-6
        
        return tp_out, fp_out, mixture_weights, mixture_locations, mixture_scales 