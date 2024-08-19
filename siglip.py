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


class SiglipVisionEmbeddings(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
    
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid" #this means that no padding is added
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2 # image size is the # of pixels and we divide that by how big is each patch
        # we raise it to the power of 2 since we have 2 dimensions
        
        self.num_positions = self.num_patches # number of positional encodings = # of patches we have
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim) # the embeddings are learned and we have num_positions number of embeddings
        # recall that each of these are added to the information extracted from the convolution 
        
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        
        _, _, height, width = pixel_values.shape ## the dimensions are batch_size, channels, height, and width
        
        patch_embeds = self.patch_embeddings(pixel_values) # get the patch embeddings from the convolution
        # [batch_size, embed_dim, num_patches_height, num_patches_width] -> [batch_size, embed_dim, num_patches]
        # note that num_patches_height = height // patch_size and num_patches_width = width // patch_size
        
        embeddings = patch_embeds.flatten(2)
        # we flatten this and get a 1-d list of patches
        
        embeddings = embeddings.transpose(1, 2)
        # [batch_size, embed_dim, num_patches] -> [batch_size, num_patches, embed_dim]
        # we transpose bc we want the number of patches to come before the embedding dimensions

        embeddings = embeddings + self.position_embeddings(self.position_ids)
        # we add the positional encodings to the embeddings
        
        return embeddings
        # [batch_size, num_patches, embed_dim]
        
class SiglipMLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size) # takes each of the embeddings and expands them into an intermediate size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size) # apply non linearity and compress it back into hidden_size
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        hidden_states = self.fc1(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_size]
        
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        
        # we use the gelu non-linearity because gelu works better than relu for this model architecture (research has shown this)
        
        hidden_states = self.fc2(hidden_states) # apply the linear layer (recompressing it back into the hidden_size)
        
        return hidden_states

class SiglipAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # this is basically 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len, _ = hidden_states.size()
        # the dimension are: [batch_size, num_patches, embed_dim]
        # we can think of the patches = sequence length
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # this basically is a transformation of the input sequence, but at this current moment, each token is independent of one another
        
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # we are splitting this according to the number of heads
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # the dimension for all the states are now: [batch_size, num_heads, num_patches, head_dim]
        
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)
        # attn_weights: [batch_size, num_heads, num_patches, num_patches]
        
        
        
        
class SiglipEncoderLayer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size # this is the embedding dimension 
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        residual = hidden_states ## save the skip connection for later
        # residual size: [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # layer normalization does not change size (linear)
        
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # self attention does not change the shape either
        
        hidden_states = hidden_states + residual # add the skip connection
        
        residual = hidden_states # save skip connection
        
        hidden_states = self.layer_norm2(hidden_states) # layer normalization
        
        hidden_states = self.mlp(hidden_states) # mlp (series of linear layers)
        
        hidden_states = residual + hidden_states
        
        return hidden_states
        
        
        
        
        
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
                 