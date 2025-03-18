
# Subgradient Method:

# The method starts at an initial guess. The algorithm then iteratively updates x by moving in the direction of the negative subgradient (to minimize the function), scaled by a learning rate alpha.
# The new point x k+1is calculated using subgradient the step size.
# The algorithm tracks the best value of the function f(x) during the iterations and keeps a history of the path taken by the subgradient updates.
# Stopping Criteria:

# The algorithm iterates for a maximum number of iterations (max_it), and typically, you would stop if the change in the function value becomes small enough (though in the code, this part is commented out for simplicity)

import numpy as np
import matplotlib.pyplot as plt

def l1_norm(x):
    return np.abs(x)

def subgradient_l1(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return np.random.uniform(-1, 1) 

def subgradient_method(f, grad_f, x_0, alpha, max_it=1000, tol=1e-6):
    x_best = x_0
    f_best = f(x_best)
    x_history = [x_best]
    f_history = [f_best]
    
    x_k = x_0
    k = 0
    
    while k < max_it:
        g_k = grad_f(x_k)
        
        x_k_next = x_k - alpha * g_k
        
        f_new = f(x_k_next)
        
        if f_new < f_best:
            f_best = f_new
            x_best = x_k_next
        
        x_history.append(x_best)
        f_history.append(f_best)
        
        # Stopping criteria: if the change in functional value is small enough
        # if np.abs(f_new - f_best) < tol:
        #     break
        
        x_k = x_k_next
        k += 1
    
    return x_best, f_best, x_history, f_history


x_0 = 5  
alpha = 0.1  
max_it = 1000  
tol = 1e-6  

x_star, f_star, x_history, f_history = subgradient_method(l1_norm, subgradient_l1, x_0, alpha, max_it, tol)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 2)
plt.plot(f_history, label="Loss (f(x))", color='blue')
plt.xlabel("Iterations")
plt.ylabel("Function Value")
plt.title("Sub-gradient Method: Loss Over Iterations")
plt.legend()

x_vals = np.linspace(-6, 6, 100)
y_vals = l1_norm(x_vals)

plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, label="L1 Norm Function", color='purple')
plt.plot(x_history, [l1_norm(x) for x in x_history], 'r-x', label='Sub-gradient Path')
plt.text(x_history[0], l1_norm(x_history[0]), 'Start', color='blue', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

plt.text(x_history[-1], l1_norm(x_history[-1]), 'End', color='green', fontsize=12, verticalalignment='top', horizontalalignment='right')

plt.scatter(x_history[-1], l1_norm(x_history[-1]), color='green', marker='x', s=100, label="End Point")  # Green 'x' for end
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Sub-gradient Method: Path on L1 Norm")
plt.legend()

plt.tight_layout()
plt.show()

print(f"Optimal x: {x_star}")
print(f"Optimal f(x): {f_star}")
