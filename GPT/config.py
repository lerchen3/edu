class TransformerConfig:
    # Model architecture
    vocab_size = 50257  # GPT-2 default
    d_model = 768      # Hidden size
    num_heads = 12
    num_layers = 12
    d_ff = 3072       # 4 * d_model is "common"
    max_seq_length = 1024
    dropout = 0.1
    
    # Training parameters
    learning_rate = 3e-4
    warmup_steps = 4000
    weight_decay = 0.01
    beta1 = 0.9 # Adam
    beta2 = 0.999 # Adam
    
    # Generation parameters
    temperature = 0.7
    top_k = 50 # number of top tokens to consider; k=1 is greedy sampling
    top_p = 0.9 # cumulative probability for top-p (nucleus) sampling