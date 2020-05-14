import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import random

x = np.linspace(0, 10, num=40)

print(x)
y = np.(x)


print(y)


def test(x, a, b):
    # return a * np.sin(b * x)
    return a*x + b

param, param_cov = curve_fit(test, x, y)
ans = (param[0] * (np.sin(param[1] * x)))

plt.plot(x, y, 'o', color='red', label="data")
plt.plot(x, ans, '--', color='blue', label="optimized data")
plt.legend()
plt.show()
