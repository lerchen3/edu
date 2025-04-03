import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Self-attention projections
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Split into multiple heads
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        
        # Apply causal mask
        if mask is None:
            causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
            causal_mask = causal_mask.to(scores.device)
            scores = scores.masked_fill(causal_mask, float('-inf'))
        else:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)
        
        # Concatenate all heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(out)
