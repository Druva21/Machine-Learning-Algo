# Iteration: For each iteration:

# Compute the gradient of the objective function at the current position.
# Set the search direction as the negative of the gradient (descent direction).
# Use backtracking line search with the Armijo rule to find an optimal step size (alpha) that ensures sufficient decrease in the function value.
# Update the position by moving along the search direction scaled by the step size.
# Stopping Criteria: The algorithm stops if the change in position between iterations is smaller than a given tolerance (default 1e-6) or if the maximum number of iterations is reached.



import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_func(x):
    df_dx1 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    df_dx2 = 200 * (x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2])

def backtracking_with_armijo(func, grad_func, x, d, alpha=1, beta=0.8, c1=1e-4):
    while func(x + alpha * d) > func(x) + c1 * alpha * np.dot(grad_func(x), d):
        alpha *= beta
    return alpha

def gradient_descent_with_exact_line_search(func, grad_func, start, tol=1e-6, max_iters=1000):
    x = np.array(start, dtype=float)
    history = [x]
    for _ in range(max_iters):
        grad = grad_func(x)
        d = -grad
        alpha = backtracking_with_armijo(func, grad_func, x, d)
        x_new = x + alpha * d
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
        history.append(x)
    return np.array(history)

start = [-1.5, 1.5]
history_exact_line_search = gradient_descent_with_exact_line_search(func, grad_func, start)

X1, X2 = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-1, 3, 100))
Z = func([X1, X2])

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
plt.plot(history_exact_line_search[:, 0], history_exact_line_search[:, 1], 'ro-', markersize=3)
plt.title("Gradient Descent with Exact Line Search (Armijo Backtracking)")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.colorbar()
plt.show()
