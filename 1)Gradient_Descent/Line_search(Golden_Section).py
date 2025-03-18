# Golden Section Search Algorithm
# 1. Initialize an interval [a, b] where the minimum of the function lies.
# 2. Calculate two intermediate points, x1 and x2, within the interval based on the golden ratio.
# 3. Evaluate the function at x1 and x2.
# 4. Narrow the interval by comparing function values at x1 and x2:
#    - If f(x1) < f(x2), update the interval to [a, x2].
#    - If f(x1) >= f(x2), update the interval to [x1, b].
# 5. Repeat steps 2-4 until the interval is smaller than a tolerance level, tol.
# 6. Return the midpoint of the final interval as the best estimate of the minimum.

import numpy as np
import matplotlib.pyplot as plt

def golden_section_search(func, a, b, tol=1e-5):
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1, f2 = func(x1), func(x2)
    while abs(b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + resphi * (b - a)
            f1 = func(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = b - resphi * (b - a)
            f2 = func(x2)
    return (a + b) / 2

def func_1d(x):
    return (x - 2)**2 + 1

a, b = -5, 5
x_opt = golden_section_search(func_1d, a, b)

x = np.linspace(a, b, 400)
y = func_1d(x)

plt.plot(x, y, label="f(x)")
plt.axvline(x=x_opt, color='r', linestyle='--', label="Golden Section Minimum")
plt.title("Golden Section Search")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
