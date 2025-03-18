# EWMA (Exponentially Weighted Moving Average): In this method, the gradients at each iteration are updated with an exponential decay factor (controlled by the parameter beta). The gradients are exponentially weighted so that the most recent gradients have more influence on the direction of optimization. The update step is based on the weighted sum of previous gradients.

# WMA (Weighted Moving Average): This method keeps a fixed window of the most recent gradients, and at each step, it calculates the simple average of the gradients within this window to determine the update direction.

# Gradient Calculation: At each step, the gradients of the Rosenbrock function with respect to x and y are calculated. These gradients guide the optimization direction.


import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def grad_rosenbrock_x(x, y, a=1, b=100):
    return -2 * (a - x) - 4 * b * x * (y - x**2)

def grad_rosenbrock_y(x, y, b=100):
    return 2 * b * (y - x**2)

def ewma_rosenbrock(initial_x, initial_y, learning_rate=0.1, beta=0.9, max_iterations=1000, clip_value=5):
    x, y = initial_x, initial_y
    v_x, v_y = 0, 0  
    x_history = []
    y_history = []
    loss_history = []
    x_history.append(initial_x)
    y_history.append(initial_y)
    
    for iteration in range(max_iterations):
       
        grad_x = grad_rosenbrock_x(x, y)
        grad_y = grad_rosenbrock_y(x, y)
        
        v_x = beta * v_x + (1 - beta) * grad_x
        v_y = beta * v_y + (1 - beta) * grad_y
        
        x -= learning_rate * v_x
        y -= learning_rate * v_y
        
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

def wma_rosenbrock(initial_x, initial_y, learning_rate=0.1, window_size=5, max_iterations=1000, clip_value=5):
    x, y = initial_x, initial_y
    gradients_x = []
    gradients_y = []
    x_history = []
    y_history = []
    loss_history = []
    x_history.append(initial_x)
    y_history.append(initial_y)
    
    for iteration in range(max_iterations):
        
        grad_x = grad_rosenbrock_x(x, y)
        grad_y = grad_rosenbrock_y(x, y)

        gradients_x.append(grad_x)
        gradients_y.append(grad_y)
        if len(gradients_x) > window_size:
            gradients_x.pop(0)
            gradients_y.pop(0)
        
        wma_grad_x = np.mean(gradients_x)
        wma_grad_y = np.mean(gradients_y)
        
        x -= learning_rate * wma_grad_x
        y -= learning_rate * wma_grad_y
        
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
learning_rate = 0.01
beta = 0.9
window_size = 10  
max_iterations = 1000

x_history_ewma, y_history_ewma, loss_history_ewma, iterations_ewma = ewma_rosenbrock(initial_x, initial_y, learning_rate, beta, max_iterations)
x_history_wma, y_history_wma, loss_history_wma, iterations_wma = wma_rosenbrock(initial_x, initial_y, learning_rate, window_size, max_iterations)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 2)
plt.plot(loss_history_ewma, label="Loss (EWMA)", color='blue')
plt.plot(loss_history_wma, label="Loss (WMA)", color='red')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("EWMA vs WMA: Loss Over Iterations")
plt.legend()

x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-1, 6, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock(X, Y)

plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 50), cmap='viridis')
plt.plot(x_history_ewma, y_history_ewma, 'r-x', label='EWMA Path', markersize=4)
plt.plot(x_history_wma, y_history_wma, 'g-o', label='WMA Path', markersize=4)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"EWMA vs WMA for Rosenbrock")
plt.text(x_history_ewma[0], y_history_ewma[0], 'Start (EWMA)', color='blue', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
plt.text(x_history_wma[0], y_history_wma[0], 'Start (WMA)', color='red', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

plt.text(x_history_ewma[-1], y_history_ewma[-1], 'End (EWMA)', color='green', fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.text(x_history_wma[-1], y_history_wma[-1], 'End (WMA)', color='purple', fontsize=12, verticalalignment='top', horizontalalignment='left')

plt.scatter(x_history_ewma[-1], y_history_ewma[-1], color='green', marker='x', s=100, label="End (EWMA)")  # Green 'x' for EWMA end
plt.scatter(x_history_wma[-1], y_history_wma[-1], color='purple', marker='x', s=100, label="End (WMA)")  # Purple 'x' for WMA end

plt.legend()

plt.tight_layout()
plt.show()
