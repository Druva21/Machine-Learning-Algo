# Damped Newton Method: This method is quite sophisticated as it uses both the gradient and the Hessian matrix to determine the optimal step size. However, to avoid overshooting or unstable updates, it incorporates a damping factor that adjusts the step size, making the method more controlled. This results in more precise convergence, though it may require more computation due to the Hessian matrix.

# Quasi-Newton Method: On the other hand, the Quasi-Newton method uses the gradient to update the solution, but it doesn’t require the Hessian matrix. Instead, it approximates it, which simplifies the process and reduces computational cost. This method is faster in many cases but may not always converge as quickly as the Damped Newton method, especially for more complex or highly non-linear functions.

# In the plots, you’ll notice that the Damped Newton path tends to be more direct and focused, likely converging faster to the optimal solution, while the Quasi-Newton path might appear a bit more gradual and spread out, reflecting the more approximate nature of its approach.

# In practical terms:

# Use Damped Newton if accuracy and a stable, reliable convergence are essential, and computational cost isn’t a primary concern.
# Choose Quasi-Newton when you need a faster solution, especially when dealing with larger, more complex problems where the Hessian matrix would be too expensive to compute.



import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0]**2 + x[1]**2  # Example function: f(x, y) = x^2 + y^2

def grad_f(x):
    return np.array([2 * x[0], 2 * x[1]]) 

def hessian_f(x):
    return np.array([[2, 0], [0, 2]]) 

def damped_newton_method(f, grad_f, hessian_f, x0, epsilon=1e-6, max_iter=100, damping=0.5):
    x_k = x0
    k = 0
    x_history = [x_k.copy()]  
    
    while np.linalg.norm(grad_f(x_k)) > epsilon and k < max_iter:
        gradient_k = grad_f(x_k)
        hessian_k = hessian_f(x_k)
        
        delta = -np.linalg.solve(hessian_k, gradient_k)
        x_k = x_k + damping * delta  
        
        x_history.append(x_k.copy())
        k += 1
    
    return x_k, x_history

def quasi_newton_method(f, grad_f, x0, epsilon=1e-6, max_iter=100, alpha=0.1):
    x_k = x0
    k = 0
    x_history = [x_k.copy()] 
    
    while np.linalg.norm(grad_f(x_k)) > epsilon and k < max_iter:
        gradient_k = grad_f(x_k)
        
        x_k = x_k - alpha * gradient_k
        x_history.append(x_k.copy())
        
        k += 1
    
    return x_k, x_history

x0 = np.array([5.0, 5.0]) 
epsilon = 1e-6  
alpha = 0.1  
damping = 0.5  

x_opt_newton, x_history_newton = damped_newton_method(f, grad_f, hessian_f, x0, epsilon, damping=damping)
x_opt_quasi, x_history_quasi = quasi_newton_method(f, grad_f, x0, epsilon, alpha=alpha)

x_history_newton = np.array(x_history_newton)
x_history_quasi = np.array(x_history_quasi)

x_vals = np.linspace(-6, 6, 100)
y_vals = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 + Y**2 

plt.figure(figsize=(10, 8))

plt.contour(X, Y, Z, levels=20, cmap='viridis')

plt.plot(x_history_newton[:, 0], x_history_newton[:, 1], 'b-o', label="Damped Newton Path", markersize=4)

plt.plot(x_history_quasi[:, 0], x_history_quasi[:, 1], 'r-x', label="Quasi-Newton Path", markersize=4)

plt.plot(x_history_newton[0, 0], x_history_newton[0, 1], 'bo', markersize=8, label="Start Point (Newton)")
plt.plot(x_history_newton[-1, 0], x_history_newton[-1, 1], 'gx', markersize=8, label="End Point (Newton)")
plt.plot(x_history_quasi[0, 0], x_history_quasi[0, 1], 'ro', markersize=8, label="Start Point (Quasi-Newton)")
plt.plot(x_history_quasi[-1, 0], x_history_quasi[-1, 1], 'mx', markersize=8, label="End Point (Quasi-Newton)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Damped Newton vs Quasi-Newton Optimization Paths")
plt.legend()
plt.grid()
plt.show()

print("Damped Newton Method Optimal solution:", x_opt_newton)
print("Function value at Damped Newton Method optimal solution:", f(x_opt_newton))
print("Quasi-Newton Method Optimal solution:", x_opt_quasi)
print("Function value at Quasi-Newton Method optimal solution:", f(x_opt_quasi))
