import numpy as np
from network import MLP
from operations import mse_loss
from tensor import Tensor
from optimizers import Adam

# Create synthetic data - let's do a simple binary classification problem
def generate_data(n_samples=1000):
    # Generate two circular clusters
    np.random.seed(42)
    
    # First cluster
    r1 = np.random.normal(0, 1, n_samples//2)
    theta1 = np.random.uniform(0, 2*np.pi, n_samples//2)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    
    # Second cluster
    r2 = np.random.normal(3, 1, n_samples//2)
    theta2 = np.random.uniform(0, 2*np.pi, n_samples//2)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    
    # Combine data
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])]).T
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    return X, y

def train_epoch(model: MLP, X: np.ndarray, y: np.ndarray, optimizer):
    # Zero gradients at start of each batch
    optimizer.zero_grad()
    
    # Forward pass
    output = model(Tensor(X, requires_grad=False))
    
    # Compute loss (MSE)
    target = Tensor(y.reshape(1, -1), requires_grad=False)
    loss = mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Use optimizer instead of manual update
    optimizer.step()
            
    return loss.data

def evaluate(model, X, y):
    output = model(Tensor(X)).data
    predictions = (output > 0.5).astype(float)
    accuracy = np.mean(predictions == y)
    return accuracy

# Generate data
X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(200)

# Create model
model = MLP(in_features=2, hidden_features=[64, 32], out_features=1)
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
n_epochs = 1000
for epoch in range(n_epochs):
    loss = train_epoch(model, X_train, y_train, optimizer)
    
    if epoch % 10 == 0:
        train_acc = evaluate(model, X_train, y_train)
        test_acc = evaluate(model, X_test, y_test)
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# Final evaluation
train_acc = evaluate(model, X_train, y_train)
test_acc = evaluate(model, X_test, y_test)
print("\nFinal Results:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Optional: Visualize the decision boundary
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[0].min() - 1, X[0].max() + 1
    y_min, y_max = X[1].min() - 1, X[1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    Z = model(Tensor(np.c_[xx.ravel(), yy.ravel()].T)).data
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[0], X[1], c=y, alpha=0.8)
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(X_train, y_train, model)