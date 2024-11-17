from tensor import Tensor
from context import Context
import numpy as np

class Function:
    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Forward pass of the operation
        Args:
            ctx: Context object to save information for backward pass
            *args: Input tensors
            **kwargs: Optional parameters
        """
        raise NotImplementedError
        
    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass to compute gradients
        Args:
            ctx: Context object containing saved tensors
            *grad_outputs: Upstream gradients
        Returns:
            Tuple of gradients for each input
        """
        raise NotImplementedError
        
    @classmethod
    def apply(cls, *args, **kwargs):
        """Execute the function"""
        ctx = Context()
        # Run forward pass
        result = cls.forward(ctx, *args, **kwargs)
        # Create output tensor
        output = Tensor(result, requires_grad=True)
        # Save input tensors for backward pass
        output.prev_tensors = args  # Store input tensors
        # Save backward function
        def _backward_function(*grad_outputs):
            return cls.backward(ctx, *grad_outputs)
        output.grad_fn = _backward_function
        return output

class MatMulFunction(Function):
    @staticmethod
    def forward(ctx, w, x):
        """
        w: weight matrix of shape (out_features, in_features)
        x: input matrix of shape (in_features, batch_size)
        """
        ctx.save_for_backward(w, x)
        return w.data @ x.data
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output shape: (out_features, batch_size)
        """
        w, x = ctx.saved_tensors
        # grad wrt w: grad_output @ x.T -> (out_features, batch_size) @ (batch_size, in_features)
        grad_w = grad_output @ x.data.T
        # grad wrt x: w.T @ grad_output -> (in_features, out_features) @ (out_features, batch_size)
        grad_x = w.data.T @ grad_output
        return grad_w, grad_x

class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.data.clip(min=0)
        
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (x.data > 0)

class AddFunction(Function):
    @staticmethod
    def forward(ctx, x, b):
        """Forward pass of Add
        Args:
            x: Input tensor of shape (features, batch_size)
            b: Bias vector of shape (features,)
        """
        ctx.save_for_backward(x, b)
        # Explicitly broadcast b to match x's shape
        b_broadcasted = np.broadcast_to(b.data[:, None], x.data.shape)
        return x.data + b_broadcasted
        
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of Add
        For x + b:
        dx = grad_output
        db = sum(grad_output, axis=1) for broadcasting
        """
        x, b = ctx.saved_tensors # (for consistency; not actually needed)
        # Sum gradients across batch dimension for bias
        grad_b = np.sum(grad_output, axis=1)
        return grad_output, grad_b

class MSELossFunction(Function):
    @staticmethod
    def forward(ctx, pred, target):
        """
        Forward pass of MSE Loss
        Args:
            pred: Predicted values tensor
            target: Target values tensor
        """
        ctx.save_for_backward(pred, target)
        diff = pred.data - target.data
        return np.mean(diff ** 2)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of MSE Loss
        The gradient of MSE with respect to predictions is:
        d(MSE)/d(pred) = 2(pred - target)/n
        where n is the number of elements
        """
        pred, target = ctx.saved_tensors
        n = pred.data.size  # total number of elements
        grad_pred = 2 * (pred.data - target.data) / n
        return (grad_pred * grad_output,  # grad wrt predictions 
                None)  # grad wrt targets (not needed)

# The actual operation functions that users call
def matmul(w: Tensor, x: Tensor) -> Tensor:
    """Matrix multiplication of input tensor with weights.
    
    Args:
        w: Weight matrix of shape (out_features, in_features)
        x: Input tensor of shape (in_features, batch_size)
    Returns:
        Tensor: Resulting tensor of shape (out_features, batch_size)
    """
    return MatMulFunction.apply(w, x)

def relu(x: Tensor) -> Tensor:
    """Rectified Linear Unit activation function.
    
    Args:
        x: Input tensor of shape (features, batch_size)
    Returns:
        Tensor: Resulting tensor of shape (features, batch_size)
    """
    return ReLUFunction.apply(x)

def add(x: Tensor, b: Tensor) -> Tensor:
    """Add a bias vector to input tensor.
    
    Args:
        x: Input tensor of shape (features, batch_size)
        b: Bias vector of shape (features,)
    Returns:
        Tensor: Resulting tensor of shape (features, batch_size)
    """
    return AddFunction.apply(x, b)

def linear(w: Tensor, x: Tensor, b: Tensor) -> Tensor:
    """Linear transformation of input tensor.
    
    Args:
        w: Weight matrix of shape (out_features, in_features)
        x: Input tensor of shape (in_features, batch_size)
        b: Bias vector of shape (out_features,)
    Returns:
        Tensor: Resulting tensor of shape (out_features, batch_size)
    """
    return add(matmul(w, x), b)

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error Loss.

    Args:
        pred: Predicted values tensor
        target: Target values tensor
    Returns:
        Tensor: Resulting tensor of shape (1,)
    """
    return MSELossFunction.apply(pred, target)
