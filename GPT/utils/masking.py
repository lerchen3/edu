import torch

def create_padding_mask(seq, pad_token_id):
    """Create mask for padding tokens"""
    return (seq != pad_token_id).unsqueeze(1).unsqueeze(2)

def create_causal_mask(size):
    """Create mask for autoregressive decoding"""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask.unsqueeze(0)

def create_combined_mask(seq, pad_token_id):
    """Combine padding mask with causal mask for decoder"""
    pad_mask = create_padding_mask(seq, pad_token_id)
    causal_mask = create_causal_mask(seq.size(1))
    return pad_mask & causal_mask 