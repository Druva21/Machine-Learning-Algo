
# Gradient and Hessian:

# The gradient of the Rosenbrock function (rosenbrock_grad(x)) provides the first-order information about how the function changes with respect to each parameter.
# The Hessian (rosenbrock_hess(x)) provides second-order information, capturing the curvature of the function at each point. This is used to determine the step direction and size for the optimization.
# Trust Region: The algorithm operates within a trust region around the current solution. The trust region is a region within which the model (approximated by the quadratic function) is assumed to be a good fit to the actual objective function.

# The size of this region is controlled by a radius 
# 𝑟
# r, which adapts dynamically based on the quality of the step taken at each iteration.
# Dogleg Approach:

# The dogleg method combines two possible step directions:
# Cauchy Point: A step derived from a simple steepest descent approach (using only the gradient).
# Newton Point: A step based on the Newton method using both the gradient and the Hessian.
# The algorithm chooses a step that lies between these two points, depending on the size of the trust region.
# Adaptive Radius: The radius of the trust region is updated at each iteration:

# If the step taken is not good (low value of rho, a measure of how well the model approximated the objective), the radius is reduced.
# If the step is good, the radius is increased, allowing for larger steps in future iterations.
# Stopping Criteria: The optimization stops when:

# The gradient becomes sufficiently small (meaning the algorithm has converged to a local minimum).
# The maximum number of iterations is reached.


import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[0] = -400.0 * x[0] * (x[1] - x[0]**2) - 2.0 * (1 - x[0])
    for i in range(1, len(x)-1):
        grad[i] = 200.0 * (x[i] - x[i-1]**2) - 400.0 * x[i] * (x[i+1] - x[i]**2) - 2.0 * (1 - x[i])
    grad[-1] = 200.0 * (x[-1] - x[-2]**2)
    return grad

def rosenbrock_hess(x):
    n = len(x)
    H = np.zeros((n, n))
    H[0, 0] = 1200 * x[0]**2 - 400 * x[1] + 2
    H[0, 1] = -400 * x[0]
    for i in range(1, n-1):
        H[i, i-1] = -400 * x[i]
        H[i, i] = 1200 * x[i]**2 - 400 * x[i+1] + 2
        H[i, i+1] = -400 * x[i]
    H[-1, -2] = -400 * x[-1]
    H[-1, -1] = 200
    return H

def trust_region_dogleg(rosenbrock, grad, hess, x0, r0, eta, epsilon=1e-6, max_iter=100):
    x = x0
    r = r0
    k = 0
    
    x_history = []
    f_history = []
    r_history = []  
    
    while np.linalg.norm(grad(x)) > epsilon and k < max_iter:
       
        g = grad(x)
        H = hess(x)
        
        p_sd = -np.linalg.inv(H).dot(g)
        
        p_nt = -np.linalg.inv(H).dot(g)
        
        if np.linalg.norm(p_nt) <= r:
            p = p_nt
        elif np.linalg.norm(p_sd) >= r:
            p = -r * g / np.linalg.norm(g)
        else:
            tau = (r - np.linalg.norm(p_sd)) / np.linalg.norm(p_nt - p_sd)
            p = p_sd + tau * (p_nt - p_sd)
        
        m_k = lambda p: rosenbrock(x) + g.T.dot(p) + 0.5 * p.T.dot(H).dot(p)
        rho = (rosenbrock(x) - rosenbrock(x + p)) / (m_k(np.zeros_like(p)) - m_k(p))
        
        if rho < 0.25:
            r = 0.25 * r
        elif rho > 0.75 and np.linalg.norm(p) == r:
            r = min(2 * r, r0)
        
        if rho >= eta:
            x = x + p
        
        x_history.append(x)
        f_history.append(rosenbrock(x))
        r_history.append(r)  
        
        k += 1
    
    return x, np.array(x_history), np.array(f_history), np.array(r_history)

x0 = np.array([1.5, 1.5])
r0 = 1.0
eta = 0.1

x_opt, x_history, f_history, r_history = trust_region_dogleg(rosenbrock, rosenbrock_grad, rosenbrock_hess, x0, r0, eta)

x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(x, y)
Z = np.array([rosenbrock(np.array([xi, yi])) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
Z = Z.reshape(X.shape)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, 50, cmap='viridis')

plt.plot(x_history[:, 0], x_history[:, 1], marker='o', color='r', label="Optimization Path")

for i in range(len(x_history)):
    circle = plt.Circle((x_history[i, 0], x_history[i, 1]), r_history[i], color='b', fill=False, linestyle='--')
    plt.gca().add_artist(circle)

plt.scatter(x_opt[0], x_opt[1], color='b', label="Optimized Point")
plt.title("Trust Region with Dogleg Optimization - Contour Plot")
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.colorbar(label='Objective function value')
plt.show()

print(f"Optimized solution: {x_opt}")
