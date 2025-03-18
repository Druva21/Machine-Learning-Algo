# How the Algorithm Works
# Initialization: We start with initial values of x=−2 and y=2, aiming to find values of x and y  that minimize the Rosenbrock function.
# Gradient Calculation: In each iteration, we compute the gradient of the Rosenbrock function with respect to x and y
# Momentum and Variance Adjustments: Adam uses an adaptive learning rate by calculating "first" (mean) and "second" (variance) moment estimates of the gradient to adjust the updates for x and y.
# Bias Correction: The algorithm corrects these moment estimates to avoid bias in the early stages of training.
# Parameter Update: x and y are updated by moving against the gradient, adjusted by the learning rate and the corrected moments.

# Stopping Criteria
# Convergence Check: The algorithm calculates the Rosenbrock function (loss) value at each step and stops if the change in loss between two consecutive iterations is very small . This indicates that the values of x and y are approaching a stable point.
# Iteration Limit: To avoid indefinite running, the maximum number of iterations is set to10 million so the algorithm will stop after reaching this limit if it hasn't converged earlier.


import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def grad_rosenbrock_x(x, y, a=1, b=100):
    return -2 * (a - x) - 4 * b * x * (y - x**2)

def grad_rosenbrock_y(x, y, b=100):
    return 2 * b * (y - x**2)

def adam_rosenbrock(initial_x, initial_y, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iterations=1000, clip_value=5):
    x, y = initial_x, initial_y
    m_x, m_y = 0, 0  
    v_x, v_y = 0, 0  
    x_history = []
    y_history = []
    loss_history = []
    x_history.append(initial_x)
    y_history.append(initial_y)
    
    for iteration in range(max_iterations):
        grad_x = grad_rosenbrock_x(x, y)
        grad_y = grad_rosenbrock_y(x, y)
        
        m_x = beta1 * m_x + (1 - beta1) * grad_x
        m_y = beta1 * m_y + (1 - beta1) * grad_y
        v_x = beta2 * v_x + (1 - beta2) * grad_x**2
        v_y = beta2 * v_y + (1 - beta2) * grad_y**2
        
        m_x_hat = m_x / (1 - beta1**(iteration + 1))
        m_y_hat = m_y / (1 - beta1**(iteration + 1))
        v_x_hat = v_x / (1 - beta2**(iteration + 1))
        v_y_hat = v_y / (1 - beta2**(iteration + 1))
        
        x -= learning_rate * m_x_hat / (np.sqrt(v_x_hat) + epsilon)
        y -= learning_rate * m_y_hat / (np.sqrt(v_y_hat) + epsilon)
        
        x = np.clip(x, -clip_value, clip_value)
        y = np.clip(y, -clip_value, clip_value)
        
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
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.99
epsilon = 1e-8
max_iterations = 10**7

x_history, y_history, loss_history,iterations = adam_rosenbrock(initial_x, initial_y, learning_rate, beta1, beta2, epsilon, max_iterations=max_iterations)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 2)
plt.plot(loss_history, label="Loss (Rosenbrock)", color='blue')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Adam: Loss Over Iterations")
plt.legend()

x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock(X, Y)

plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 50), cmap='viridis')
plt.plot(x_history, y_history, 'r-x', label='Adam Path', markersize=4)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Adams for Rosenbrock Function(iterations={iterations})")
plt.text(x_history[0], y_history[0], 'Start', color='blue', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

plt.text(x_history[-1], y_history[-1], 'End', color='green', fontsize=12, verticalalignment='top', horizontalalignment='left')

plt.scatter(x_history[-1], y_history[-1], color='green', marker='x', s=100, label="End Point") 
plt.legend()

plt.tight_layout()
plt.show()
