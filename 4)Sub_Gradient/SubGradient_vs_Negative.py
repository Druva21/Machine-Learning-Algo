
# Subgradient (blue vector):

# Think of it as the direction of steepest ascent. When you take a step in this direction, you’re moving uphill along the function, trying to increase the value of the function the most.
# At the origin (0,0), for ∣x∣+∣y∣, this would represent the direction where the function is increasing. It’s like standing at the corner of a city intersection and deciding which street to walk on to climb uphill.
# Negative Subgradient (red vector):

# On the flip side, the negative subgradient is the direction of steepest descent—the direction you’d walk if you wanted to decrease the function's value as quickly as possible. It’s like choosing the street that leads downhill, away from the peak.
# At the origin, walking in the negative subgradient direction means you’re moving towards the valley where the function’s value is lower.
# Key Differences:
# The subgradient pushes you uphill, and the negative subgradient takes you downhill. In optimization, you usually follow the negative subgradient to minimize the function (like finding the lowest point in a valley).
# The vectors show you the immediate direction you should take in a situation where the function is not smooth, like at the corner of a sharp peak or valley (like at x=0 for the absolute value function).
# So, the subgradient method helps you navigate tricky spots (like sharp corners) by guiding you toward a better point, either by climbing up or going down, depending on the direction you choose.

import numpy as np
import matplotlib.pyplot as plt

def absolute_value(x):
    return np.abs(x)

def subgradient(x):
    if x < 0:
        return -1 
    elif x > 0:
        return 1  
    else:
        return np.random.uniform(-1, 1) 
    
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = absolute_value(X) + absolute_value(Y) 
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=np.linspace(0, 5, 50), cmap='viridis')
plt.colorbar(label='Function Value')
plt.title("Contour Plot of |x| + |y|")

x0, y0 = 0, 0

grad_x = subgradient(x0)
grad_y = subgradient(y0)

plt.quiver(x0, y0, grad_x, grad_y, color='blue', angles='xy', scale_units='xy', scale=0.1, label='Subgradient (0,0)')
plt.quiver(x0, y0, -grad_x, -grad_y, color='red', angles='xy', scale_units='xy', scale=0.1, label='Negative Subgradient (0,0)')

plt.text(x0 + 0.1, y0 + 0.1, 'Subgradient', color='blue', fontsize=12)
plt.text(x0 - 0.2, y0 - 0.2, 'Negative Subgradient', color='red', fontsize=12)

plt.scatter(x0, y0, color='black', s=100, zorder=5)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()
