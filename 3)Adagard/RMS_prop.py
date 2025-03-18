# Optimization with RMSprop:

# The algorithm computes the gradients of the Rosenbrock function with respect to x and y at each iteration.
# The squared gradients are accumulated using a weighted average controlled by the parameter beta.
# RMSprop updates the x and y values by adjusting them with the scaled gradients, taking into account the accumulated squared gradients. This helps adjust the learning rate dynamically for each parameter, making it more robust for non-stationary problems.
# The values of x and y are clipped to a predefined range to avoid overflow or excessive values.
# Stopping Criteria:

# The algorithm continues iterating until a convergence condition is met, which is based on the change in loss between consecutive iterations being less than a small threshold (1e-6).
# The iteration count is tracked, and the history of the parameter updates (x_history, y_history) and the loss function (loss_history) are recorded.


import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def grad_rosenbrock_x(x, y, a=1, b=100):
    return -2 * (a - x) - 4 * b * x * (y - x**2)

def grad_rosenbrock_y(x, y, b=100):
    return 2 * b * (y - x**2)

def rmsprop_rosenbrock(initial_x, initial_y, learning_rate=0.001, beta=0.9, epsilon=1e-8, max_iterations=1000, clip_value=5):
    x, y = initial_x, initial_y
    grad_sq_x, grad_sq_y = 0, 0
    x_history = []
    y_history = []
    loss_history = []
    x_history.append(initial_x)
    y_history.append(initial_y)
    
    for iteration in range(max_iterations):
        grad_x = grad_rosenbrock_x(x, y)
        grad_y = grad_rosenbrock_y(x, y)
        
        grad_sq_x = beta * grad_sq_x + (1 - beta) * grad_x**2
        grad_sq_y = beta * grad_sq_y + (1 - beta) * grad_y**2
        
        x -= learning_rate * grad_x / (np.sqrt(grad_sq_x) + epsilon)
        y -= learning_rate * grad_y / (np.sqrt(grad_sq_y) + epsilon)
        
        x = np.clip(x, -clip_value, clip_value)
        y = np.clip(y, -clip_value, clip_value)
        
        x_history.append(x)
        y_history.append(y)
        
        loss = rosenbrock(x, y)
        loss_history.append(loss)
        
        if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < 1e-6:
            print(f"Converged at iteration {iteration}")
            break
    
    return x_history, y_history, loss_history, iteration

initial_x = -2
initial_y = 2
learning_rate = 0.001
beta = 0.9
epsilon = 1e-8
max_iterations = 10**3

x_history, y_history, loss_history, iterations = rmsprop_rosenbrock(initial_x, initial_y, learning_rate, beta, epsilon, max_iterations=max_iterations)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 2)
plt.plot(loss_history, label="Loss (Rosenbrock)", color='blue')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("RMSprop: Loss Over Iterations")
plt.legend()

x_vals = np.linspace(-2.1, -1.3, 100)
y_vals = np.linspace(1.5, 2.5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock(X, Y)

plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 50), cmap='viridis')
plt.plot(x_history, y_history, 'r-x', label='RMSprop Path', markersize=4)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"RMSprop for Rosenbrock Function(iterations={iterations})")
plt.text(x_history[0], y_history[0], 'Start', color='blue', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

plt.text(x_history[-1], y_history[-1], 'End', color='green', fontsize=12, verticalalignment='top', horizontalalignment='left')

plt.scatter(x_history[-1], y_history[-1], color='green', marker='x', s=100, label="End Point")
plt.legend()

plt.tight_layout()
plt.show()
