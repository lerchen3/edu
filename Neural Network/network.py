from module import Module
from layers import Linear, ReLU, Sequential

class MLP(Module):
    def __init__(self, in_features: int, hidden_features: list[int], out_features: int):
        super().__init__()
        
        layers = []
        prev_features = in_features
        
        # Create hidden layers
        for hidden_size in hidden_features:
            layers.append(Linear(prev_features, hidden_size))
            layers.append(ReLU())
            prev_features = hidden_size
            
        # Add output layer
        layers.append(Linear(prev_features, out_features))
        
        # Use register_module explicitly
        self.register_module('model', Sequential(*layers))

    def forward(self, x):
        return self.model(x)
