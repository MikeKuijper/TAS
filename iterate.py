import sys
import time
import numpy as np
import scipy
import pyarrow
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

rolling_window = 20
fit_degree = 5


def diff(lst):
    return lst[1]-lst[0]


df = pd.read_csv("test2.csv")
# df['dt'] = df["time"].rolling(window=2).apply(lambda s: diff(s)/1e6, raw=True)
# df['d_gyroADC[0]'] = df["gyroADC[0]"].rolling(window=2).apply(lambda s: diff(s), raw=True)
# df['d_gyroADC[1]'] = df["gyroADC[1]"].rolling(window=2).apply(lambda s: diff(s), raw=True)
# df['d_gyroADC[2]'] = df["gyroADC[2]"].rolling(window=2).apply(lambda s: diff(s), raw=True)
#
# df['dv_gyroADC[0]'] = df["d_gyroADC[0]"] / df["dt"]
# df['dv_gyroADC[1]'] = df["d_gyroADC[1]"] / df["dt"]
# df['dv_gyroADC[2]'] = df["d_gyroADC[2]"] / df["dt"]
#
# df['md_gyroADC[0]'] = df.rolling(window=rolling_window)['dv_gyroADC[0]'].mean()
# df['md_gyroADC[1]'] = df.rolling(window=rolling_window)['dv_gyroADC[1]'].mean()
# df['md_gyroADC[2]'] = df.rolling(window=rolling_window)['dv_gyroADC[2]'].mean()

# df['m_gyroADC[0]'] = df.rolling(window=rolling_window)['gyroADC[0]'].mean()
# df['m_gyroADC[1]'] = df.rolling(window=rolling_window)['gyroADC[1]'].mean()
# df['m_gyroADC[2]'] = df.rolling(window=rolling_window)['gyroADC[2]'].mean()

T = df['time'].values.reshape(-1, 1) / 1e6
omega_x = df['gyroADC[0]'].values.reshape(-1, 1)
omega_y = df['gyroADC[1]'].values.reshape(-1, 1)
omega_z = df['gyroADC[2]'].values.reshape(-1, 1)

romega_x = make_pipeline(PolynomialFeatures(degree=fit_degree), LinearRegression(fit_intercept = True))
romega_x.fit(T, omega_x)
romega_dot_x = np.polyder(np.flip(romega_x.named_steps.linearregression.coef_).reshape(-1))

romega_y = make_pipeline(PolynomialFeatures(degree=fit_degree), LinearRegression(fit_intercept = True))
romega_y.fit(T, omega_y)
romega_dot_y = np.polyder(np.flip(romega_y.named_steps.linearregression.coef_).reshape(-1))

romega_z = make_pipeline(PolynomialFeatures(degree=fit_degree), LinearRegression(fit_intercept = True))
romega_z.fit(T, omega_z)
romega_dot_z = np.polyder(np.flip(romega_z.named_steps.linearregression.coef_).reshape(-1))

T = np.arange(min(T), max(T), 0.01).reshape(-1, 1)

plt.plot(T, romega_x.predict(T), color="gray", linestyle="dashed")
# plt.plot(T, np.polyval(romega_dot_x, X))
# plt.plot(df["time"]/1e6, df["gyroADC[0]"])

plt.plot(T, romega_y.predict(T), color="gray", linestyle="dashed")
# plt.plot(T, np.polyval(romega_dot_y, X))
# plt.plot(df["time"]/1e6, df["gyroADC[1]"])
#
plt.plot(T, romega_z.predict(T), color="gray", linestyle="dashed")
# plt.plot(T, np.polyval(romega_dot_z, X))
# plt.plot(df["time"]/1e6, df["gyroADC[2]"])

def create_I(x):
    return np.array([[x[0], x[1], x[3]],
                     [x[1], x[2], x[4]],
                     [x[3], x[4], x[5]]])


def iterate(omega_1, omega_dot_1, omega_2, omega_dot_2):
    def it(x):
        I = create_I(x)
        r_1 = np.matmul(I, omega_dot_1) + np.cross(omega_1, np.matmul(I, omega_1))
        r_2 = np.matmul(I, omega_dot_2) + np.cross(omega_2, np.matmul(I, omega_2))
        return np.array([r_1, r_2]).reshape(-1)

    X = scipy.optimize.fsolve(it, np.ones(6))
    M = it(X)
    print(M)
    # error = np.matmul(M.reshape(1, -1), M.reshape(-1, 1)).reshape(-1)[0]
    # print("Error: ", error)
    # print(it(X))
    return create_I(X)


def iterate_2(df, omega_0):
    x_0 = np.zeros(6)
    def it(x):
        omega = omega_0
        # omega_dot = omega_dot_0
        error = np.zeros(6)
        t_0 = df['time'].values[0]
        t_end = df['time'].values[-1]
        t_diff = t_end-t_0
        i = 0
        for r in df.rolling(window=2):
            if len(r) < 2:
                i += 1
                continue
            _I = create_I(x)

            t_1 = r['time'].values[0] / 1e6
            t_2 = r['time'].values[1] / 1e6
            # if i == 1:
            #     global x_0
            #     x_0 = iterate(np.array([romega_x.predict(t_1.reshape(1, -1)),
            #                             romega_y.predict(t_1.reshape(1, -1)),
            #                             romega_z.predict(t_1.reshape(1, -1))]).reshape(-1),
            #                   np.array([np.polyval(romega_dot_x, t_1),
            #                             np.polyval(romega_dot_y, t_1),
            #                             np.polyval(romega_dot_z, t_1)]).reshape(-1),
            #                   np.array([romega_x.predict(t_2.reshape(1, -1)),
            #                             romega_y.predict(t_2.reshape(1, -1)),
            #                             romega_z.predict(t_2.reshape(1, -1))]).reshape(-1),
            #                   np.array([np.polyval(romega_dot_x, t_2),
            #                             np.polyval(romega_dot_y, t_2),
            #                             np.polyval(romega_dot_z, t_2)]).reshape(-1))
            #     print(x_0)
            #     print(np.linalg.det(x_0))
            dt = t_2 - t_1
            ddt = dt / 150
            omega_dot = None
            for t in np.arange(t_1, t_2, ddt):
                omega_dot = np.matmul(np.linalg.inv(_I), -np.cross(omega, np.matmul(_I, omega)))
                # print(omega_dot)
                omega = omega + omega_dot * ddt
            omega_error = omega - np.array([romega_x.predict(t_2.reshape(1, -1)),
                                            romega_y.predict(t_2.reshape(1, -1)),
                                            romega_z.predict(t_2.reshape(1, -1))]).reshape(-1)
            omega_dot_error = omega_dot - np.array([np.polyval(romega_dot_x, t_2),
                                                    np.polyval(romega_dot_y, t_2),
                                                    np.polyval(romega_dot_z, t_2)]).reshape(-1)
            error += abs(np.array([omega_error, omega_dot_error])).reshape(-1)
            # if t_2 >= t_0 + t_diff:
            #     break
        return error
    X = scipy.optimize.fsolve(it, np.array([1, 0.1, 1, -0.1, 0, 1]))
    M = it(X)

    error = np.matmul(M.reshape(1, -1), M.reshape(-1, 1)).reshape(-1)[0]
    # print(error)
    print("Error: ", error)
    # print(it(X))
    return create_I(X)

omega_0 = np.array([df["gyroADC[0]"].values[0],
                    df["gyroADC[1]"].values[0],
                    df["gyroADC[2]"].values[0]])
# # I = iterate_2(df, omega_0)
# print(I)


A = []
inertiaMatrix = None
inertiaMatrix_0 = None
inertias = []
        # x = np.linalg.solve(np.matmul(a.T, a), np.zeros(6))
start_time = time.time()
for r in df.rolling(window=1):
    t = r['time'].values[0] / 1e6
    o = np.array([romega_x.predict(t.reshape(1, -1)),
                  romega_y.predict(t.reshape(1, -1)),
                  romega_z.predict(t.reshape(1, -1))]).reshape(-1)
    o_dot = np.array([np.polyval(romega_dot_x, t),
                      np.polyval(romega_dot_y, t),
                      np.polyval(romega_dot_z, t)]).reshape(-1)

    line_x = [o_dot[0], -o[2] * o[0] + o_dot[1], -o[2] * o[1], o[1] * o[0] + o_dot[2], o[1] ** 2 - o[2] ** 2, o[1] * o[2]]
    A.append(line_x)
    line_y = [o[2] * o[0], o[2] * o[1] + o_dot[0], o_dot[1], o[2] ** 2 - o[0] ** 2, -o[0] * o[1] + o_dot[2], -o[0] * o[2]]
    A.append(line_y)
    line_z = [-o[1] * o[0], o[0] ** 2 - o[1] ** 2, o[0] * o[1], -o[1] * o[2] + o_dot[0], o[0] * o[2] + o_dot[1], o_dot[2]]
    A.append(line_z)
    # print(A)
    if len(A) >= 6:
        a = np.array(A).reshape(-1, 6)
        b = np.zeros(len(A)).T
        # x, r, R, s = np.linalg.lstsq(a.astype('float'), b.astype('float'), rcond=-1)
        eigen_values, eigen_vectors = np.linalg.eig(np.matmul(a.T, A))
        x = eigen_vectors[:, eigen_values.argmin()]
        inertias.extend(x)
        global I
        inertiaMatrix = create_I(x)
        if len(A) == 6*15:
            inertiaMatrix_0 = create_I(x)
            print("Test took %s seconds" % (time.time() - start_time))
print("Full took %s seconds" % (time.time() - start_time))

# sys.exit()

# Verification
omega = omega_0
omega_f = omega_0
X = []
Y = []
Z = []
X_f = []
Y_f = []
Z_f = []
Time = []
print("I =", inertiaMatrix)
for r in df.rolling(window=2):
    if len(r) < 2:
        continue
    t_1 = r['time'].values[0] / 1e6
    t_2 = r['time'].values[1] / 1e6
    dt = t_2 - t_1
    ddt = dt/1000
    i = 0
    inv = np.linalg.inv(inertiaMatrix)
    inv_f = np.linalg.inv(inertiaMatrix_0)
    for t in np.arange(t_1, t_2, ddt):
        omega_dot = np.matmul(inv, -np.cross(omega, np.matmul(inertiaMatrix, omega)))
        omega = omega + omega_dot * ddt
        omega_dot_f = np.matmul(inv_f, -np.cross(omega_f, np.matmul(inertiaMatrix_0, omega_f)))
        omega_f = omega_f + omega_dot_f * ddt
        # if i % 100 == 0:
    X.append(omega[0])
    Y.append(omega[1])
    Z.append(omega[2])
    X_f.append(omega_f[0])
    Y_f.append(omega_f[1])
    Z_f.append(omega_f[2])
    Time.append(t_1)
        # i += 1

Time = np.array(Time)
plt.plot(Time, X)
plt.plot(Time, Y)
plt.plot(Time, Z)

# plt.plot(Time, X_f)
# plt.plot(Time, Y_f)
# plt.plot(Time, Z_f)
plt.show()

# error = []
# for i in range(len(inertias)//6):
#     e = 0
#     for j in range(1, 7):
#         e += abs(inertias[0::j][i] - inertias[0::j][-1])
#     error.append(e)

# ax = plt.figure().add_subplot(111)
# # print(inertias[0::6])
# plt.plot(inertias[0::6])
# plt.plot(inertias[1::6])
# plt.plot(inertias[2::6])
# plt.plot(inertias[3::6])
# plt.plot(inertias[4::6])
# plt.plot(inertias[5::6])
# # plt.plot(error)
# ax.set_yscale('log')
# plt.show()