import torch
import torch.nn as nn
import math
from .encoder import EncoderLayer
from .decoder import DecoderLayer
from ..utils.positional_encoding import positional_encoding
from ..utils.initialization import initialize_weights
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Token embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = positional_encoding(config.max_seq_length, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.embed_norm = nn.LayerNorm(config.d_model)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Output layer
        self.final_layer = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize weights
        self.apply(initialize_weights)
        
    def encode(self, src, src_mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:src.size(1)].to(src.device)
        x = self.embed_dropout(self.embed_norm(x))
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        return x
        
    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:tgt.size(1)].to(tgt.device)
        x = self.embed_dropout(self.embed_norm(x))
        
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, memory, src_mask, tgt_mask)
        return x
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.final_layer(decoder_output)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Generate new tokens after the given context."""
        for _ in range(max_new_tokens):
            # Crop context if it exceeds max_seq_length
            idx_cond = idx if idx.size(1) <= self.config.max_seq_length else idx[:, -self.config.max_seq_length:]
            
            # Forward pass
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # Only take the last time step
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            idx = torch.cat((idx, next_token), dim=1)
            
        return idx
