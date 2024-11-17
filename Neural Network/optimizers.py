import numpy as np
from tensor import Tensor

class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, max_grad_norm=5.0):
        """Adam optimizer w/ grad clipping.
        
        Args:
            parameters: Iterable of Tensor objects containing parameters to optimize
            lr: Learning rate (default: 0.001)
            betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
            eps: Term added to denominator to improve numerical stability (default: 1e-8)
            weight_decay: Weight decay (L2 penalty) (default: 0)
            max_grad_norm: Maximum norm for gradient clipping (default: 5.0)
        """
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
        # Initialize momentum and velocity states
        self.state = {
            'step': 0,
            'm': [np.zeros_like(p.data) for p in self.parameters],  # First moment estimates
            'v': [np.zeros_like(p.data) for p in self.parameters]   # Second moment estimates
        }
    
    def clip_gradients(self):
        """Clips gradients by global norm."""
        # Calculate global norm of all gradients
        total_norm = 0
        for p in self.parameters:
            if p.grad is not None:
                param_norm = np.linalg.norm(p.grad)
                total_norm += param_norm ** 2
        total_norm = np.sqrt(total_norm)
        
        # Apply clipping
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.parameters:
                if p.grad is not None:
                    p.grad *= clip_coef

    def step(self):
        """Performs a single optimization step."""
        self.state['step'] += 1
        
        # Clip gradients
        if self.max_grad_norm > 0:
            self.clip_gradients()
        
        # Bias correction terms: want, in expectation, 1 - beta^t = (1-beta)(1+beta+beta^2+...+beta^(t-1))
        bias_correction1 = 1 - self.beta1 ** self.state['step']
        bias_correction2 = 1 - self.beta2 ** self.state['step']
        
        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                grad = p.grad
                
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * p.data
                
                # Update momentum (first moment estimate)
                self.state['m'][i] = self.beta1 * self.state['m'][i] + (1 - self.beta1) * grad
                
                # Update velocity (second moment estimate)
                self.state['v'][i] = (self.beta2 * self.state['v'][i] + 
                                    (1 - self.beta2) * np.square(grad))
                
                # Bias-corrected estimates
                m_hat = self.state['m'][i] / bias_correction1
                v_hat = self.state['v'][i] / bias_correction2
                
                # Update parameters
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Sets gradients of all parameters to None."""
        for p in self.parameters:
            p.grad = None