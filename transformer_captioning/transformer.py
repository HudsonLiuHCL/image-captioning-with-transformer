# Credit to the CS-231n course at Stanford, from which this assignment is adapted
import numpy as np
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        N, S, D = query.shape
        N, T, D = value.shape     

        Q = self.query_proj(query)      
        K = self.key_proj(key)          
        V = self.value_proj(value)     

        dot_product = Q @ K.transpose(1, 2)   
        dot_product = dot_product / math.sqrt(D)    


        if attn_mask is not None:

            additive_mask = (attn_mask == 0).float() * (-1e9)
            dot_product = dot_product + additive_mask

        attn = F.softmax(dot_product, dim=-1)       # (N, S, T)
        attn = self.dropout(attn)
        y = attn @ V                                # (N, S, D)

        return y


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        N, S, D = query.shape
        N, T, D = value.shape
        H = self.num_heads
        Hd = self.head_dim

        Q = self.q_proj(query).reshape(N, S, H, Hd).transpose(1, 2)
        K = self.k_proj(key).reshape(N, T, H, Hd).transpose(1, 2)
        V = self.v_proj(value).reshape(N, T, H, Hd).transpose(1, 2)

        dot_product = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Hd)

        if attn_mask is not None:
            additive_mask = (attn_mask == 0).float() * -1e9
            dot_product = dot_product + additive_mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(dot_product, dim=-1)
        attn = self.dropout(attn)

        y = torch.matmul(attn, V)
        y = y.transpose(1, 2).reshape(N, S, D)
        return self.out_proj(y)



class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.encoding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, S, D = x.shape
        positions = torch.arange(S, device=x.device).unsqueeze(0)  
        pos_embed = self.encoding(positions)                     
        output = x + pos_embed
        return self.dropout(output)



class SelfAttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, seq, mask):
        attn_out = self.self_attn(seq, seq, seq, attn_mask=mask)
        out = self.layernorm(seq + self.dropout(attn_out))
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, seq, cond):
        attn_out = self.cross_attn(seq, cond, cond)
        out = self.norm(seq + self.dropout(attn_out))
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, seq):
        out = self.mlp(seq)
        out = self.norm(seq + self.dropout(out))
        return out


class DecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        self.self_atn_block = SelfAttentionBlock(input_dim, num_heads, dropout)
        self.cross_atn_block = CrossAttentionBlock(input_dim, num_heads, dropout)
        self.feedforward_block = FeedForwardBlock(input_dim, num_heads, dim_feedforward, dropout)

    def forward(self, seq, cond, mask):
        out = self.self_atn_block(seq, mask)
        out = self.cross_atn_block(out, cond)
        return self.feedforward_block(out)
       
class TransformerDecoder(nn.Module):
    def __init__(self, word_to_idx, idx_to_word, input_dim, embed_dim, num_heads=4,
                 num_layers=2, max_length=50, device = 'cuda'):
        """
        Construct a new TransformerDecoder instance.
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension of input image feature vectors.
        - embed_dim: Embedding dimension of the transformer.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self.idx_to_word = idx_to_word
        
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads) for _ in range(num_layers)])
        
        self.caption_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_length)
        self.feature_embedding = nn.Linear(input_dim, embed_dim)
        self.score_projection = nn.Linear(embed_dim, vocab_size) 

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)

    def get_data_embeddings(self, features, captions):
        feature_embedding = self.feature_embedding(features).unsqueeze(1)
        caption_embedding = self.caption_embedding(captions)
        caption_embedding = self.positional_encoding(caption_embedding)
        return feature_embedding, caption_embedding

    def get_causal_mask(self, _len):
        mask = torch.tril(torch.ones(_len, _len, device=self.device))
        return mask

                                      
    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.
        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)
        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        features_embed, captions_embed = self.get_data_embeddings(features, captions)
        mask = self.get_causal_mask(captions_embed.shape[1])
        mask.to(captions_embed.dtype)
        
        output = captions_embed
        for layer in self.layers:
            output = layer(output, features_embed, mask=mask)

        scores = self.score_projection(output)
        return scores

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.
        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length
        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features).to(self.device)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).to(self.device)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.cpu().numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions


