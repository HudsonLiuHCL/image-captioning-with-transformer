import sys
import torch.nn as nn
import torch

sys.path.append("../transformer_captioning") 
from transformer import (
    AttentionLayer,
    MultiHeadAttentionLayer,
    PositionalEncoding,
    SelfAttentionBlock,
    CrossAttentionBlock,
    FeedForwardBlock
)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attention = SelfAttentionBlock(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForwardBlock(d_model, num_heads, d_ff, dropout=dropout)

    def forward(self, seq, mask):
        x = self.self_attention(seq, mask)
        x = self.feed_forward(x)

        return x

class ViT(nn.Module):


    def __init__(self, patch_dim, d_model, d_ff, num_heads, num_layers, num_patches, num_classes, device = 'cuda'):


        super().__init__()

        self.patch_dim = patch_dim
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.device = device

        self.patch_embedding = nn.Linear(patch_dim * patch_dim * 3, d_model)

        self.positional_encoding = PositionalEncoding(d_model, dropout=0.1)

        self.fc = nn.Linear(d_model, num_classes)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)

    def patchify(self, images):


        N, C, H, W = images.shape

        patches = images.unfold(2, self.patch_dim, self.patch_dim).unfold(3, self.patch_dim, self.patch_dim)
        patches = patches.contiguous().view(N, C, -1, self.patch_dim, self.patch_dim)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous()
        patches = patches.view(N, -1, self.patch_dim * self.patch_dim * C)
        
        return patches

    def forward(self, images):

        N = images.shape[0]
        
        patches = self.patchify(images)
        patches_embedded = self.patch_embedding(patches)
        
        cls_tokens = self.cls_token.expand(N, -1, -1) 
        output = torch.cat([cls_tokens, patches_embedded], dim=1)  

        output = self.positional_encoding(output)

        mask = None

        for layer in self.layers:
            output = layer(output, mask)


        cls_output = output[:, 0, :]  
        logits = self.fc(cls_output)  

        return logits

    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)