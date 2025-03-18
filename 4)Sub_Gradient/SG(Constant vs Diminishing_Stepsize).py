# Subgradient Method:

# The algorithm starts at an initial point as [5,5] and iteratively updates the solution by taking a step in the direction opposite to the subgradient.
# There are two variations for the step size:
# Fixed Step Size: The step size remains constant throughout the iterations.
# Diminishing Step Size: The step size decreases over time, proportional to 1/(k+1), where k is the iteration index.
# Iteration Process:

# At each iteration:
# The subgradient at the current point is computed.
# A step is taken in the opposite direction of the subgradient, with the appropriate step size.
# The algorithm records the new point if it results in a lower function value.
# The algorithm stops when the difference between the new and previous points is below a predefined tolerance (tol), or when the maximum number of iterations (max_it) is reached.


import numpy as np
import matplotlib.pyplot as plt

def l1_norm_2d(x):
    return np.abs(x[0]) + np.abs(x[1])

def subgradient_l1_2d(x):
    grad = np.zeros_like(x)
    grad[0] = 1 if x[0] > 0 else (-1 if x[0] < 0 else np.random.uniform(-1, 1))
    grad[1] = 1 if x[1] > 0 else (-1 if x[1] < 0 else np.random.uniform(-1, 1))
    return grad

def subgradient_method_2d(f, grad_f, x_0, alpha, max_it=1000, tol=1e-6, diminishing=False):
    x_best = x_0
    f_best = f(x_best)
    x_history = [x_best.copy()]
    
    x_k = x_0
    k = 0
    
    while k < max_it:
        g_k = grad_f(x_k)
        
        step_size = alpha / (k + 1) if diminishing else alpha
        
        x_k_next = x_k - step_size * g_k
        
        if f(x_k_next) < f_best:
            f_best = f(x_k_next)
            x_best = x_k_next
        
        x_history.append(x_k_next.copy())
        
        if np.linalg.norm(x_k_next - x_k) < tol:
            break
        
        x_k = x_k_next
        k += 1
    
    return x_best, f_best, x_history

x_0 = np.array([5.0, 5.0])  
alpha = 0.5  # Step size
max_it = 100  
tol = 1e-6  

_, _, x_history_fixed = subgradient_method_2d(l1_norm_2d, subgradient_l1_2d, x_0, alpha, max_it, tol, diminishing=False)
_, _, x_history_diminishing = subgradient_method_2d(l1_norm_2d, subgradient_l1_2d, x_0, alpha, max_it, tol, diminishing=True)

x_history_fixed = np.array(x_history_fixed)
x_history_diminishing = np.array(x_history_diminishing)

x_vals = np.linspace(-6, 6, 100)
y_vals = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.abs(X) + np.abs(Y)

plt.figure(figsize=(12, 6))

plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
plt.plot(x_history_fixed[:, 0], x_history_fixed[:, 1], 'b-o', label="Fixed Step Path", markersize=4)
plt.plot(x_history_diminishing[:, 0], x_history_diminishing[:, 1], 'orange', marker='x', label="Diminishing Step Path", markersize=4)

plt.plot(x_history_fixed[0, 0], x_history_fixed[0, 1], 'bo', markersize=8, label="Start Point")
plt.plot(x_history_fixed[-1, 0], x_history_fixed[-1, 1], 'bx', markersize=10, label="Fixed End Point")
plt.plot(x_history_diminishing[-1, 0], x_history_diminishing[-1, 1], 'x', color='orange', markersize=10, label="Diminishing End Point")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Sub-gradient Method: Contours of L1 Norm with Fixed vs. Diminishing Step Sizes")
plt.legend()
plt.grid()
plt.show()
