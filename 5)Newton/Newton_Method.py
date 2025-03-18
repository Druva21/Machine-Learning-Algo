# Gradient and Hessian Calculation: At each iteration, the gradient and Hessian of the function are calculated at the current point. The gradient indicates the direction of steepest ascent, while the Hessian provides information about the curvature of the function.

# Damping: To ensure numerical stability and avoid issues with non-positive-definite Hessians (which might arise in some optimization problems), the Hessian is modified by adding a damping factor (λ) if needed.

# Update Step: The update step uses the equation:cx=h*cf where h is the inverse of the Hessian matrix (or its damped version) and cf is the gradient. The point is updated by adding this step to the current point.

# Convergence Check: The algorithm continues to iterate until either the gradient is small enough (indicating that the point is close to a local minimum) or the maximum number of iterations is reached.


import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return np.array([2 * x[0], 2 * x[1]])

def hessian_f(x):
    return np.array([[2, 0], [0, 2]])

def damped_newton_method(f, grad_f, hessian_f, x0, epsilon=1e-6, max_iter=100, lambda_factor=1e-3):
    x_k = x0
    k = 0
    x_history = [x_k.copy()]
    
    while np.linalg.norm(grad_f(x_k)) > epsilon and k < max_iter:
        hessian_k = hessian_f(x_k)
        gradient_k = grad_f(x_k)
        
        try:
            np.linalg.cholesky(hessian_k)
            delta = np.linalg.solve(hessian_k, -gradient_k)
        except np.linalg.LinAlgError:
            hessian_k_damped = hessian_k + lambda_factor * np.eye(len(x_k))
            delta = np.linalg.solve(hessian_k_damped, -gradient_k)
        
        x_k = x_k + delta
        x_history.append(x_k.copy())
        k += 1
    
    return x_k, x_history

x0 = np.array([5.0, 5.0])
epsilon = 1e-6
lambda_factor = 1e-3

x_opt, x_history = damped_newton_method(f, grad_f, hessian_f, x0, epsilon, lambda_factor=lambda_factor)

x_history = np.array(x_history)

x_vals = np.linspace(-6, 6, 100)
y_vals = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 + Y**2

plt.figure(figsize=(8, 6))

plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.plot(x_history[:, 0], x_history[:, 1], 'r-o', label="Damped Newton Path", markersize=4)

plt.plot(x_history[0, 0], x_history[0, 1], 'bo', markersize=8, label="Start Point")
plt.plot(x_history[-1, 0], x_history[-1, 1], 'gx', markersize=8, label="End Point")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Damped Newton Method Optimization Path")
plt.legend()
plt.grid()
plt.show()

print("Optimal solution:", x_opt)
print("Function value at optimal solution:", f(x_opt))
