# Both methods aim to optimize the Rosenbrock function but use different approaches. The Newton method leverages the exact Hessian matrix, giving it the potential to converge more quickly by incorporating curvature information. However, it can be computationally intensive as it requires calculating and inverting the Hessian. On the other hand, BFGS approximates the Hessian iteratively, making it less computationally demanding. BFGS is often more stable and handles large-scale problems more effectively, even if it converges slightly slower than Newton's method due to the approximation.

# Overall, Newton’s method converges faster with exact curvature but requires more computational resources, whereas BFGS offers a balance between speed and computational efficiency.



import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(x, y, a=1, b=100):
    dfdx = -2*(a - x) - 4*b*x*(y - x**2)
    dfdy = 2*b*(y - x**2)
    return np.array([dfdx, dfdy])

def rosenbrock_hessian(x, y, a=1, b=100):
    d2fdx2 = 2 - 4*b*y + 12*b*x**2
    d2fdy2 = 2*b
    d2fdxdy = -4*b*x
    return np.array([[d2fdx2, d2fdxdy], [d2fdxdy, d2fdy2]])

def bfgs_optimizer(f, grad_f, x0, epsilon=1e-6, max_iter=50):
    x = x0
    g = grad_f(*x)
    H = np.eye(len(x))
    path = [x.copy()]

    for k in range(max_iter):
        if np.linalg.norm(g) < epsilon:
            break
        
        p = -H @ g
        alpha = 0.001
        x_new = x + alpha * p
        g_new = grad_f(*x_new)
        
        delta_x = x_new - x
        delta_g = g_new - g
        
        rho = 1.0 / (delta_g.T @ delta_x)
        H = (np.eye(len(x)) - rho * np.outer(delta_x, delta_g)) @ H @ (np.eye(len(x)) - rho * np.outer(delta_g, delta_x)) + rho * np.outer(delta_x, delta_x)

        x = x_new
        g = g_new
        path.append(x.copy())

    return np.array(path), x

def newton_optimizer(f, grad_f, hessian_f, x0, epsilon=1e-6, max_iter=50):
    x = x0
    g = grad_f(*x)
    path = [x.copy()]

    for k in range(max_iter):
        if np.linalg.norm(g) < epsilon:
            break
        
        H = hessian_f(*x)
        p = -np.linalg.inv(H) @ g
        alpha = 0.001
        x_new = x + alpha * p
        g_new = grad_f(*x_new)

        x = x_new
        g = g_new
        path.append(x.copy())

    return np.array(path), x

x_range = np.linspace(0.5, 1.6, 100)
y_range = np.linspace(1, 2, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

x0 = np.array([1.5, 1.5])

bfgs_path, bfgs_min = bfgs_optimizer(rosenbrock, rosenbrock_grad, x0, max_iter=1000)
newton_path, newton_min = newton_optimizer(rosenbrock, rosenbrock_grad, rosenbrock_hessian, x0, max_iter=1000)

plt.figure(figsize=(10, 8))
cp = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='jet')
plt.colorbar(cp)

plt.plot(bfgs_path[:, 0], bfgs_path[:, 1], linestyle='-', color='r', marker='o', markersize=8, linewidth=1, label='BFGS Path')
plt.plot(newton_path[:, 0], newton_path[:, 1], linestyle='-', color='b', marker='s', markersize=8, linewidth=1, label='Newton Path')

plt.scatter(bfgs_min[0], bfgs_min[1], color='r', zorder=1, label='BFGS Minimum')
plt.scatter(newton_min[0], newton_min[1], color='b', zorder=1, label='Newton Minimum')

plt.title('BFGS vs Newton Optimization on Rosenbrock Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
