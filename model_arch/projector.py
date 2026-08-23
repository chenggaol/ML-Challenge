#input shape = [B,1,model_dim]
#output shape = [B,1,960]
"""
Projector layer
Purpose: Take the embedding from sensor encoder and move it to the SmolLM's 960 embedding space
What it learns:
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


#linear

@dataclass
class LinearProjectorConfig:
    model_dim: int = 128
    llm_dim: int = 960


class LinearProjector(nn.Module):
    def __init__(self, config: LinearProjectorConfig = LinearProjectorConfig()):
        super().__init__()
        self.project = nn.Linear(config.model_dim, config.llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, model_dim]
        return self.project(x)  # [B, 1, llm_dim]


#mlp

@dataclass
class MLPProjectorConfig:
    model_dim: int = 128
    hidden_dim: int = 512
    llm_dim: int = 960
    dropout: float = 0.1


class MLPProjector(nn.Module):
    def __init__(self, config: MLPProjectorConfig = MLPProjectorConfig()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.model_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, model_dim]
        return self.net(x)  # [B, 1, llm_dim]
