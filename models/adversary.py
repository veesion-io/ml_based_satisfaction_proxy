#!/usr/bin/env python3
"""
Adversary model for the precision-aware training network
"""

import torch
import torch.nn as nn

class Adversary(nn.Module):
    """
    Adversary model to predict the total number of alerts from the latent representation.
    The goal is to make the main model's representation invariant to the number of alerts.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass for the adversary.
        x: (batch_size, input_dim) - Latent representation from the main model
        """
        return self.network(x) 