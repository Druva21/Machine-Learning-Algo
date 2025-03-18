# Line Search: For each step, find a small multiplier alpha for d so moving along d minimizes the function. Adjust alpha if it gets too large.
# Position Update: Move to the new position with x_new = x + alpha * d.
# Gradient Update: Calculate the new gradient g_new at x_new.
# Direction Update: Set the next direction as d_new = -g_new + beta * d, where beta adjusts for the shape of the function and allows efficient movement.

# Small Gradient: Stop if the size of the gradient |g| is below a tiny number, like 1e-6, meaning we are close to a minimum.

# Max Iterations: Stop if the loop hits a maximum number of tries (like 1000), to avoid running too long.

import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(x, y, a=1, b=100):
    dfdx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dfdy = 2 * b * (y - x**2)
    return np.array([dfdx, dfdy])

def line_search(f, x, d, alpha=1, rho=0.5, c=1e-4):
    while f(*(x + alpha * d)) > f(*x) + c * alpha * np.dot(rosenbrock_grad(*x), d):
        alpha *= rho
    return alpha

def conjugate_gradient_optimizer(f, grad_f, x0, epsilon=1e-6, max_iter=1000):
    x = x0
    g = grad_f(*x)
    d = -g
    path = [x.copy()]

    for k in range(max_iter):
        if np.linalg.norm(g) < epsilon:
            break
        
        alpha = line_search(f, x, d)
        x_new = x + alpha * d
        g_new = grad_f(*x_new)

        beta = (np.linalg.norm(g_new) ** 2) / (np.linalg.norm(g) ** 2)
        d = -g_new + beta * d

        x = x_new
        g = g_new
        path.append(x.copy())

    return np.array(path), x

x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

x0 = np.array([-2, 2])

cg_path, cg_min = conjugate_gradient_optimizer(rosenbrock, rosenbrock_grad, x0, max_iter=1000)

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
plt.plot(cg_path[:, 0], cg_path[:, 1], linestyle='-', color='green', marker='o', markersize=4, linewidth=1, label='Conjugate Gradient Path')
plt.scatter(cg_min[0], cg_min[1], color='green', s=80, label='Conjugate Gradient Minimum', edgecolor='black')

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Conjugate Gradient Optimization Path on Rosenbrock Function")
plt.show()
