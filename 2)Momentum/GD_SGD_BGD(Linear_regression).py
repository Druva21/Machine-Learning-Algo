
# Gradient Descent (GD):
# GD is like walking steadily towards your destination, using the entire map (your whole dataset) to decide your next step. Each step is calculated carefully based on all available information, which makes the journey smooth and predictable. However, it can be slow if you have a big map (a large dataset) because you must check every detail before moving forward. It’s stable, but may take longer.

# Stochastic Gradient Descent (SGD):
# SGD is like taking quick steps, but instead of using the entire map to decide your path, you only check a small part of it (one data point at a time). This makes your journey faster, but a bit more erratic. You might find yourself zigzagging because you're reacting to small, random pieces of the map, but over time, you get closer to your destination more quickly. It’s faster but more unpredictable.

# Batch Gradient Descent (BGD):
# BGD combines the best of both worlds. It looks at a small chunk of the map (a batch of data points) instead of the entire map but does so more consistently than SGD. It’s like reviewing a handful of clues at a time instead of rushing through one or looking at everything all at once. It can balance speed and stability, providing a more controlled path towards the destination without the noise of SGD.

# In practical terms, GD is great for small datasets, SGD is ideal when speed is a priority and the dataset is large, and BGD works well when you want a mix of speed and stability.




import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 100
d = 2
X = np.random.rand(N, d) * 10
Y = 4 * X[:, 0] + 3 * X[:, 1] + 7 + np.random.randn(N) * 2

X_augmented = np.c_[np.ones((N, 1)), X]

learning_rate = 0.001
num_iterations = 100
batch_size = 10

W_gd = np.random.randn(d + 1)
W_sgd = np.random.randn(d + 1)
W_bgd = np.random.randn(d + 1)

W_gd_history = [W_gd.copy()]
W_sgd_history = [W_sgd.copy()]
W_bgd_history = [W_bgd.copy()]

for _ in range(num_iterations):
    predictions = X_augmented @ W_gd
    errors = predictions - Y
    gradient = (2 / N) * X_augmented.T @ errors
    W_gd -= learning_rate * gradient
    W_gd_history.append(W_gd.copy())

for _ in range(num_iterations):
    for i in range(N):
        xi = X_augmented[i:i+1]
        yi = Y[i:i+1]
        prediction = xi @ W_sgd
        error = prediction - yi
        gradient = 2 * xi.T @ error
        W_sgd -= learning_rate * gradient
    W_sgd_history.append(W_sgd.copy())

for _ in range(num_iterations):
    indices = np.random.choice(N, batch_size, replace=False)
    X_batch = X_augmented[indices]
    Y_batch = Y[indices]
    predictions = X_batch @ W_bgd
    errors = predictions - Y_batch
    gradient = (2 / batch_size) * X_batch.T @ errors
    W_bgd -= learning_rate * gradient
    W_bgd_history.append(W_bgd.copy())

W_gd_history = np.array(W_gd_history)
W_sgd_history = np.array(W_sgd_history)
W_bgd_history = np.array(W_bgd_history)

W0_range = np.linspace(-5, 9, 100)
W1_range = np.linspace(0.5, 10, 100)
W0, W1 = np.meshgrid(W0_range, W1_range)

Z = np.zeros_like(W0)
W2_fixed = 3

for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        W_temp = np.array([W0[i, j], W1[i, j], W2_fixed])
        predictions = X_augmented @ W_temp
        errors = predictions - Y
        Z[i, j] = np.mean(errors ** 2)

plt.figure(figsize=(12, 8))
plt.contour(W0, W1, Z, levels=30, cmap="viridis")
plt.plot(W_gd_history[:, 0], W_gd_history[:, 1], 'r-o', label='GD Path', markersize=5)
plt.plot(W_sgd_history[:, 0], W_sgd_history[:, 1], 'b-s', label='SGD Path', markersize=5)
plt.plot(W_bgd_history[:, 0], W_bgd_history[:, 1], 'g-^', label='BGD Path', markersize=5)

plt.scatter(W_gd_history[0, 0], W_gd_history[0, 1], color='red', marker='x', s=100, label='GD Start')
plt.scatter(W_gd_history[-1, 0], W_gd_history[-1, 1], color='red', marker='o', s=100, label='GD End')

plt.scatter(W_sgd_history[0, 0], W_sgd_history[0, 1], color='blue', marker='x', s=100, label='SGD Start')
plt.scatter(W_sgd_history[-1, 0], W_sgd_history[-1, 1], color='blue', marker='s', s=100, label='SGD End')

plt.scatter(W_bgd_history[0, 0], W_bgd_history[0, 1], color='green', marker='x', s=100, label='BGD Start')
plt.scatter(W_bgd_history[-1, 0], W_bgd_history[-1, 1], color='green', marker='^', s=100, label='BGD End')

plt.xlabel("W0")
plt.ylabel("W1")
plt.legend()
plt.title("Optimization Paths for GD, SGD, and BGD on Linear Regression")
plt.show()
