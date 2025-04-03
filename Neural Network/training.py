import numpy as np
from tensor import Tensor
from optimizers import Adam, SOAP


def demo_adam():
    # Create two parameters for demonstration
    params = [Tensor(1.0), Tensor(2.0)]
    opt = Adam(params, lr=0.01)
    # Assign dummy gradient to parameters
    for p in params:
        p.grad = np.array(0.5)
    print("Before Adam:", [p.data for p in params])
    opt.step()
    opt.zero_grad()
    print("After Adam:", [p.data for p in params])


def demo_soap():
    # Create a 2x2 grid of parameters for SOAP
    grid = [[Tensor(1.0), Tensor(2.0)],
            [Tensor(3.0), Tensor(4.0)]]
    opt = SOAP(grid, lr=0.01)
    # Assign dummy gradient to each tensor
    for row in grid:
        for p in row:
            p.grad = np.array(0.5)
    print("Before SOAP:")
    for row in grid:
        print([p.data for p in row])
    opt.step()
    opt.zero_grad()
    print("After SOAP:")
    for row in grid:
        print([p.data for p in row])


def main():
    print("Testing Adam Optimizer")
    demo_adam()
    print("\nTesting SOAP Optimizer")
    demo_soap()


if __name__ == '__main__':
    main() 