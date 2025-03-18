# Adagrad Optimization Loop:
# For each iteration, the gradients with respect to x and y  are calculated.

# The squared gradients are then accumulated in gx and gy​, allowing Adagrad to adapt the learning rate based on the history of gradients.

# The Adagrad update rule adjusts x and y scaling the learning rate by the inverse of the square root of the accumulated gradient sums. This allows larger steps in directions with fewer accumulated gradients and smaller steps in directions with more accumulated gradients, which is particularly helpful in optimizing functions with curved, narrow valleys like the Rosenbrock function.

# Stopping Criterion: The loop continues for a maximum number of iterations (set to 10 million in this example) or until the change in the loss between iterations falls below a small threshold (1e-6), indicating convergence.


import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def grad_rosenbrock_x(x, y, a=1, b=100):
    return -2 * (a - x) - 4 * b * x * (y - x**2)

def grad_rosenbrock_y(x, y, b=100):
    return 2 * b * (y - x**2)

def adagrad_rosenbrock(initial_x, initial_y, learning_rate=0.1, epsilon=1e-8, max_iterations=1000):
    x, y = initial_x, initial_y
    g_x, g_y = 0, 0  
    x_history = []
    y_history = []
    loss_history = []
    x_history.append(initial_x)
    y_history.append(initial_y)
    
    
    for iteration in range(max_iterations):
        grad_x = grad_rosenbrock_x(x, y)
        grad_y = grad_rosenbrock_y(x, y)
        
        g_x += grad_x**2
        g_y += grad_y**2
        
        x -= (learning_rate / (np.sqrt(g_x + epsilon))) * grad_x
        y -= (learning_rate / (np.sqrt(g_y + epsilon))) * grad_y
        
        x_history.append(x)
        y_history.append(y)
        
        loss = rosenbrock(x, y)
        loss_history.append(loss)
        
        if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < 1e-6:
            print(f"Converged at iteration {iteration}")
            break
    
    return x_history, y_history, loss_history,iteration

initial_x = -2
initial_y = 2
learning_rate = 0.1
max_iterations = 10**7

x_history, y_history, loss_history,iterations = adagrad_rosenbrock(initial_x, initial_y, learning_rate, max_iterations=max_iterations)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 2)
plt.plot(loss_history, label="Loss (Rosenbrock)", color='blue')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Adagrad: Loss Over Iterations")
plt.legend()

x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock(X, Y)

plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 50), cmap='viridis')
plt.plot(x_history, y_history, 'r-x', label='Adagrad Path', markersize=4)
plt.text(x_history[0], y_history[0], 'Start', color='blue', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

plt.text(x_history[-1], y_history[-1], 'End', color='green', fontsize=12, verticalalignment='top', horizontalalignment='left')

plt.scatter(x_history[-1], y_history[-1], color='green', marker='x', s=100, label="End Point")  
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Adagrad for Rosenbrock Function(iterations:{iterations})")
plt.legend()

plt.tight_layout()
plt.show()
