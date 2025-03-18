# Gradient Descent with Momentum:

# The gradients are computed.
# The velocities are updated by considering the previous velocity and current gradients, with a momentum factor (typically ≈0.9) to smooth out updates and help escape local minima.
# The position (x,y) is updated based on these velocities.

# Stopping Criteria: The algorithm stops early if either:

# The gradients fall below a threshold, indicating a very small change in the function (meaning convergence).
# The loss (error) between iterations changes by less than a defined threshold, indicating the algorithm is no longer making significant progress.


import numpy as np
import matplotlib.pyplot as plt

def f(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def df_x(x, y, a=1, b=100):
    return -2 * (a - x) - 4 * b * x * (y - x**2)

def df_y(x, y, b=100):
    return 2 * b * (y - x**2)

def gradient_descent_momentum(initial_x, initial_y, learning_rate, momentum, max_iterations, grad_threshold=1e-4, loss_threshold=1e-6):
    x, y = initial_x, initial_y
    v_x, v_y = 0, 0
    x_history = []
    y_history = []
    loss_history = []
    x_history.append(initial_x)
    y_history.append(initial_y)
    
    for i in range(max_iterations):
        grad_x = df_x(x, y)
        grad_y = df_y(x, y)
        
        v_x = momentum * v_x - learning_rate * grad_x
        v_y = momentum * v_y - learning_rate * grad_y
        
        x += v_x
        y += v_y
        
        current_loss = f(x, y)
        
        x_history.append(x)
        y_history.append(y)
        loss_history.append(current_loss)
        
        if abs(grad_x) < grad_threshold and abs(grad_y) < grad_threshold:
            print(f"Stopped early at iteration {i} as gradient threshold reached.")
            break
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < loss_threshold:
            print(f"Stopped early at iteration {i} as loss change threshold reached.")
            break
    
    return x_history, y_history, loss_history, i + 1

initial_x = -2
initial_y = 2
learning_rate = 0.001
momentum = 0.9
max_iterations = 1000

x_history, y_history, loss_history, num_iterations = gradient_descent_momentum(initial_x, initial_y, learning_rate, momentum, max_iterations)

print('plotting')

plt.figure(figsize=(12, 5))

x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 50), cmap='viridis')
plt.plot(x_history, y_history, 'r-x', label='momentum GD', markersize=4)
plt.text(x_history[0], y_history[0], 'Start', color='blue', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
plt.text(x_history[-1], y_history[-1], 'End', color='green', fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.scatter(x_history[-1], y_history[-1], color='green', marker='x', s=100, label="End Point")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Momentum-based G.D(Rosenbrock function) - {num_iterations} iterations")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_history, label="Loss", color='red')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Over Iterations for Rosenbrock Function")
plt.legend()

plt.tight_layout()
plt.show()
