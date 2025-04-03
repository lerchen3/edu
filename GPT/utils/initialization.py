import torch.nn as nn

def initialize_weights(model):
    """Initialize weights using Xavier uniform for linear layers"""
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight)
    if hasattr(model, 'bias') and model.bias is not None:
        nn.init.constant_(model.bias, 0) 