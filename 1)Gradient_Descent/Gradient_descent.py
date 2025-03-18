# Gradient Descent Iteration:

# The algorithm begins at an initial guess (start) for the parameters .
# In each iteration, the gradient is computed at the current point, and the parameters are updated by moving in the direction opposite to the gradient (the negative gradient direction), scaled by the learning rate (alpha).
# The process continues until either the change in the parameters between iterations becomes sufficiently small (less than a tolerance, tol), indicating convergence, or the maximum number of iterations is reached.
# Convergence Criteria:

# The algorithm stops if the change in parameters between iterations is smaller than a predefined tolerance value (
# tol=1×10 −6), suggesting that the algorithm has converged to an optimal solution.
# If convergence is not achieved within the specified number of iterations (max_iters=1000), the algorithm terminates and returns the result up to that point.



import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return (1 - x[0])**2 + 10 * (x[1] - x[0]**2)**2

def grad_func(x):
    dfdx0 = -2 * (1 - x[0]) - 40 * x[0] * (x[1] - x[0]**2)
    dfdx1 = 20 * (x[1] - x[0]**2)
    return np.array([dfdx0, dfdx1])

def gradient_descent(func, grad_func, start, alpha=0.001, tol=1e-6, max_iters=1000):
    x = np.array(start, dtype=float) 
    history = [x] 
    for _ in range(max_iters):
        grad = grad_func(x) 
        x_new = x - alpha * grad  
        if np.linalg.norm(x_new - x) < tol:  
            break
        x = x_new  
        history.append(x) 
    return np.array(history)

start = [-1.5, 1.5]
history = gradient_descent(func, grad_func, start)
X1, X2 = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-1, 3, 400))
Z = func([X1, X2])

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=3) 
plt.title("Basic Gradient Descent")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.colorbar()
plt.show()
