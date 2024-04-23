import numpy as np
import math
import matplotlib.pyplot as plt


# Returns the required coefficients for an O(n-1) order accurate approximation of the derivative.
def derivativeCoefficients(n):
    T = np.zeros(n * n).reshape(n, n)
    res = np.zeros(n)
    res[1] = 1

    for y in range(n):
        for x in range(n):
            if y == 0:
                T[y, x] = 1
            elif x == 0:
                T[y, x] = 0
            else:
                T[y, x] = (-x) ** y / math.factorial(y)
    return np.linalg.solve(T, res)


h = 1
m = 8       # for m-1 order accuracy
coefficients = derivativeCoefficients(m)
print(coefficients)


def f(x):
    # return 1 / (1 + np.exp(-x))
    return x**2


X = np.arange(0, 4, 0.01)
Y = f(X)

plt.plot(X, Y)

x_0 = 2
X_test = np.array([x_0 - h * i / m for i in range(m)])
Y_test = f(X_test)
# print(x_test)
deriv = (Y_test @ coefficients.reshape(-1, 1)) / (1/m*h)
print(deriv[0])

y_grad = deriv * X + Y_test[0] - deriv * X_test[0]

plt.plot(X, y_grad, color="k")
plt.plot(X_test, Y_test, "xg")
plt.show()