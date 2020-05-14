import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import random

# x = np.linspace(0, 10, num=40)
x = np.array([1, 11, 17, 19, 21, 23, 24, 25, 28, 30, 32, 35])
# print(x)
# y = np.random(x)
y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])


# print(y)


def test(x, a, b):
    # return a * np.sin(b * x)
    return a * x + b


param, param_cov = curve_fit(test, x, y)
print(param)
print(param_cov)
ans = (param[0] * (np.array(param[1] * x)))
print(ans)
ans = np.array(param[0] * x)
plt.plot(x, y, 'o', color='red', label="data")
plt.plot(x, ans, '--', color='blue', label="optimized data")
plt.legend()
plt.show()
