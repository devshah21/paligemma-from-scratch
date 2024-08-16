from typing import Optional, Tuple
import torch
import torch.nn as nn

## the reason we use a config class is because pali-gemma comes in different sizes

class SiglipVisionConfig:
    
    def __init__(
        self,
        hidden_size=768, # size of embedding vector
        intermediate_size=3072, # size of linear layer in feedforward
        num_hidden_layers=12, # number of layers in vision transformer
        num_attention_heads=12, # number of attention heads in multi-head attention
        num_channels=3, # number of channels in input image
        image_size=224, 
        patch_size=16, # each patch is 16x16
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None, # how many output embeddings it will produce
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens
                 