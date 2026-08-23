#most important thing is input and output shape to keep compatibility, stuff in between do whatever you please

#input shape = [B,128,9]
#output shape = [B,1,embedding_dim]

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CNNConfig:
    input_channels: int = 9
    sequence_length: int = 128
    conv_channels: tuple[int, ...] = (64, 128, 256) #increase number of in channels(dim=1)
    kernel_sizes: tuple[int, ...] = (7, 5, 3) #decrease the spatial dimension(dim=2)
    strides: tuple[int, ...] = (2, 2, 2) 
    embedding_dim: int = 128
    num_classes: int = 6
    dropout: float = 0.1



class CNNEncoder(nn.Module):
    def __init__(self, config: CNNConfig = CNNConfig()):
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_channels = config.input_channels

        #constructs the 3 convolution blocks
        for out_channels, kernel_size, stride in zip(
            config.conv_channels, config.kernel_sizes, config.strides
        ):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(config.dropout))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers) #all the conv blocks combined
        self.pool = nn.AdaptiveAvgPool1d(1) #pools dim=2 into 1 by taking average
        self.project = nn.Linear(in_channels, config.embedding_dim) #project to the size of token

    def forward(self, x: torch.Tensor):
        # x: [B, input_channels, sequence_length]
        x = x.transpose(1,2)
        x = self.conv(x)#[B, C_last, L']
        x = self.pool(x).squeeze(-1) #[B, C_last]
        x = self.project(x) #[B, embedding_dim]
        return x.unsqueeze(1)# [B, 1, embedding_dim]


@dataclass
class TransformerConfig:
    input_channels: int = 9
    sequence_length: int = 128
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 256
    embedding_dim: int = 128
    num_classes: int = 6
    dropout: float = 0.1
    use_cls_token: bool = False #True = learned [CLS] token pooling, False = mean pooling
    #the pooling takes transformer layers output [B,L,d_model] and combines into one token [B,d_model]
    #LEAVING THIS OUT FOR NOW COULD BE SOMETHING IMPLEMENTED IN FUTURE(write in report)


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig = TransformerConfig()):
        super().__init__()
        self.input_embedding = nn.Linear(config.input_channels,config.d_model) #project the 9 sensor readings into more dim
        self.position_encoder = nn.Parameter(torch.zeros(config.sequence_length, config.d_model))
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model = config.d_model,
            nhead = config.num_heads,
            dim_feedforward = config.ffn_dim,
            dropout = config.dropout,
            batch_first = True #dim 0 = B
        ) 

        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers = config.num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(config.d_model,config.embedding_dim)

    def forward(self,x):
        x = self.input_embedding(x) #[B,L,d_model]
        x = x + self.position_encoder #same shape
        x = self.transformer(x) #[B,L,d_model]
        x = x.transpose(1,2) #[B,d_model,L]
        x = self.pool(x).squeeze(-1) # [B,d_model]
        x = self.project(x) # [B,embedding_dim]
        return x.unsqueeze(-2) # [B,1,embedding_dim]
        
        