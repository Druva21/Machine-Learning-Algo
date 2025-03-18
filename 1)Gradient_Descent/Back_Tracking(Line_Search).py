# Algorithm Description
# Gradient Descent Step:

# The algorithm starts at an initial point, x, in two-dimensional space and calculates the gradient at this point, which indicates the direction of steepest ascent.
# To minimize the function, it moves in the opposite direction (steepest descent) by setting the search direction, d, as the negative gradient.
# Backtracking Line Search:

# Instead of using a fixed step size, the algorithm dynamically adjusts the step size, alpha, using Backtracking Line Search.
# Starting with an initial alpha, it iteratively reduces alpha (by multiplying it with beta, a factor typically less than 1) until it satisfies the Armijo condition, ensuring sufficient decrease in function value.
# This adaptive step size helps the algorithm avoid overshooting the minimum while progressing efficiently towards it.
# Stopping Criteria
# The gradient descent process iterates until one of the following criteria is met:

# Tolerance-Based Stopping:

# The difference between consecutive positions, x and x_new, is calculated.
# If this difference falls below a small tolerance value, tol (set to 10^-6), the algorithm assumes it is close enough to a minimum and stops, indicating convergence.
# Maximum Iterations:

# The algorithm is capped by max_iters (set to 1000) to prevent infinite looping in case it fails to converge within a reasonable number of steps.


import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x):
    a = 1
    b = 100
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    a = 1
    b = 100
    dfdx1 = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    dfdx2 = 2 * b * (x[1] - x[0]**2)
    return np.array([dfdx1, dfdx2])

def backtracking_line_search(func, grad_func, x, d, alpha=1, beta=0.8, c=1e-4):
  
    while func(x + alpha * d) > func(x) + c * alpha * np.dot(grad_func(x), d):
        alpha *= beta  
    return alpha

def gradient_descent_with_backtracking(func, grad_func, start, tol=1e-6, max_iters=1000):
    x = np.array(start, dtype=float)
    history = [x]
    for _ in range(max_iters):
        grad = grad_func(x) 
        d = -grad
        alpha = backtracking_line_search(func, grad_func, x, d) 
        x_new = x + alpha * d  
        if np.linalg.norm(x_new - x) < tol:  
            break
        x = x_new  
        history.append(x) 
    return np.array(history)

start = [-2, 2]

history_bt = gradient_descent_with_backtracking(rosenbrock, rosenbrock_grad, start)

x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-1, 3, 100)
X1, X2 = np.meshgrid(x_range, y_range)
Z = rosenbrock([X1, X2])

plt.figure(figsize=(10, 6))
cp = plt.contour(X1, X2, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
plt.plot(history_bt[:, 0], history_bt[:, 1], 'go-', markersize=3, label='Optimization Path')
plt.title("Gradient Descent with Backtracking Line Search on Rosenbrock Function")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.colorbar(cp)
plt.legend()
plt.show()
