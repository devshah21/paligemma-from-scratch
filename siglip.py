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
        
class SiglipVisionTransformer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbeddings(config) # get the embeddings / patches from vision embeddings
        self.encoder = SiglipEncoder(config) # we will run it through a list of layers of transformers
        
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps) # layer normalization
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        
        hidden_states = self.embeddings(pixel_values) # convert them into embeddings (extracting patches)
        
        last_hidden_state = self.encoder(inputs_embeds=hidden_states) # the encoder is a list of layers of the transformer (multi-head attention, norm, feed-forward)
        
        last_hidden_state = self.post_layernorm(last_hidden_state) # layer normalization 
        
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values):
        # [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        # our vision transformer will convert into this output dimension
        # and it will return a list of embeddings of size embed_dim
        return self.vision_model(pixel_values=pixel_values)
                 