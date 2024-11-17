import numpy as np
from collections import OrderedDict
from module import Module
from tensor import Tensor
from operations import linear, relu

class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        # Initialize weights and bias
        weight_data = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        bias_data = np.zeros((out_features,))
        
        # Create tensors
        weight = Tensor(weight_data, requires_grad=True)
        bias = Tensor(bias_data, requires_grad=True)
        
        # Register parameters explicitly
        self.register_parameter('weight', weight)
        self.register_parameter('bias', bias)

    def forward(self, x: Tensor) -> Tensor:
        return linear(self.weight, x, self.bias)

class ReLU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x : Tensor) -> Tensor:
        return relu(x)

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.register_module(str(idx), module)

    def forward(self, x : Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x 