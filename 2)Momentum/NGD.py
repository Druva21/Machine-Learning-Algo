
# Lookahead Step: At each iteration, the algorithm first calculates the "lookahead" position by adding the momentum-scaled velocity to the current position. This lookahead step helps the algorithm anticipate future directions, improving convergence.

# Gradient Computation: The gradients of the Rosenbrock function are computed at the lookahead position, which are used to update the velocity terms.

# Velocity Update: The velocities vx and vy are updated by applying the gradients and the learning rate. The velocity is then used to update the current position of x  and y.

# Stopping Criteria: The algorithm stops early if:

# The gradients fall below a specified threshold, indicating a local minimum has been reached.
# The change in the loss function between iterations becomes smaller than a set threshold, suggesting the optimization has converged.

import numpy as np
import matplotlib.pyplot as plt

def f(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def df_x(x, y, a=1, b=100):
    return -2 * (a - x) - 4 * b * x * (y - x**2)

def df_y(x, y, b=100):
    return 2 * b * (y - x**2)

def nesterov_gradient_descent(initial_x, initial_y, learning_rate, momentum, max_iterations, grad_threshold=1e-4, loss_threshold=1e-6):
    x, y = initial_x, initial_y
    v_x, v_y = 0, 0
    x_history = []
    y_history = []
    loss_history = []
    x_history.append(initial_x)
    y_history.append(initial_y)
    
    for i in range(max_iterations):
        lookahead_x = x + momentum * v_x
        lookahead_y = y + momentum * v_y
        
        grad_x = df_x(lookahead_x, lookahead_y)
        grad_y = df_y(lookahead_x, lookahead_y)
        
        v_x = momentum * v_x - learning_rate * grad_x
        v_y = momentum * v_y - learning_rate * grad_y
        
        x += v_x
        y += v_y
        
        if np.abs(x) > 1e10 or np.abs(y) > 1e10:
            print("Overflow detected, stopping optimization.")
            break
        
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
learning_rate = 0.0001
momentum = 0.9
max_iterations = 10**7

x_history, y_history, loss_history, num_iterations = nesterov_gradient_descent(initial_x, initial_y, learning_rate, momentum, max_iterations)

plt.figure(figsize=(12, 5))

x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 50), cmap='viridis')

plt.plot(x_history, y_history, 'r-x', label='Nesterov GD', markersize=4)

plt.text(x_history[0], y_history[0], 'Start', color='blue', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

plt.text(x_history[-1], y_history[-1], 'End', color='green', fontsize=12, verticalalignment='top', horizontalalignment='left')

plt.scatter(x_history[-1], y_history[-1], color='green', marker='x', s=100, label="End Point")

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Nesterov GD Rosenbrock fn (Iterations: {num_iterations})")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_history, label="Loss (Nesterov GD)", color='red')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Over Iterations for Rosenbrock Function")
plt.legend()

plt.tight_layout()
plt.show()
