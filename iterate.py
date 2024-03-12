import time
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

fit_degree = 6      # Degree of the polynomial fit for

fig, ax = plt.subplots() # Initialise plot
df = pd.read_csv("test2.csv") # Load data file

T = df['time'].values.reshape(-1, 1) / 1e6 # Get times from the data, scaled to seconds

# Get gyroscope data and convert to rad/s
omega_x = df['gyroADC[0]'].values.reshape(-1, 1) * math.pi / 180
omega_y = df['gyroADC[1]'].values.reshape(-1, 1) * math.pi / 180
omega_z = df['gyroADC[2]'].values.reshape(-1, 1) * math.pi / 180

romega_x = make_pipeline(PolynomialFeatures(degree=fit_degree), LinearRegression(fit_intercept = True))
romega_x.fit(T, omega_x)
romega_dot_x = np.polyder(np.flip(romega_x.named_steps.linearregression.coef_).reshape(-1))

romega_y = make_pipeline(PolynomialFeatures(degree=fit_degree), LinearRegression(fit_intercept = True))
romega_y.fit(T, omega_y)
romega_dot_y = np.polyder(np.flip(romega_y.named_steps.linearregression.coef_).reshape(-1))

romega_z = make_pipeline(PolynomialFeatures(degree=fit_degree), LinearRegression(fit_intercept = True))
romega_z.fit(T, omega_z)
romega_dot_z = np.polyder(np.flip(romega_z.named_steps.linearregression.coef_).reshape(-1))

T = np.arange(np.min(T), np.max(T), 0.01).reshape(-1, 1)

plt.plot(T, romega_x.predict(T), color="gray", linestyle="dotted")
# plt.plot(T, np.polyval(romega_dot_x, X))
# plt.plot(df["time"]/1e6, df["gyroADC[0]"], color="gray", linestyle="dashed")

plt.plot(T, romega_y.predict(T), color="gray", linestyle="dotted")
# plt.plot(T, np.polyval(romega_dot_y, X))
# plt.plot(df["time"]/1e6, df["gyroADC[1]"], color="gray", linestyle="dashed")

plt.plot(T, romega_z.predict(T), color="gray", linestyle="dotted")
# plt.plot(T, np.polyval(romega_dot_z, X))
# plt.plot(df["time"]/1e6, df["gyroADC[2]"], color="gray", linestyle="dashed")

def create_I(x):
    return np.array([[x[0], x[1], x[3]],
                     [x[1], x[2], x[4]],
                     [x[3], x[4], x[5]]])

omega_0 = np.array([df["gyroADC[0]"].values[0],
                    df["gyroADC[1]"].values[0],
                    df["gyroADC[2]"].values[0]]) * math.pi / 180

A = []
inertiaMatrix = None
inertiaMatrix_0 = None
inertias = []

iterations = 100

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
    line_y = [o[2] * o[0], o[2] * o[1] + o_dot[0], o_dot[1], o[2] ** 2 - o[0] ** 2, -o[0] * o[1] + o_dot[2], -o[0] * o[2]]
    line_z = [-o[1] * o[0], o[0] ** 2 - o[1] ** 2, o[0] * o[1], -o[1] * o[2] + o_dot[0], o[0] * o[2] + o_dot[1], o_dot[2]]
    A.append(line_x)
    A.append(line_y)
    A.append(line_z)

    if len(A) >= 6:
        a = np.array(A).reshape(-1, 6)
        b = np.zeros(len(A)).T
        # x, r, R, s = np.linalg.lstsq(a.astype('float'), b.astype('float'), rcond=-1)
        eigen_values, eigen_vectors = np.linalg.eig(np.matmul(a.T, A))
        x = eigen_vectors[:, eigen_values.argmin()]
        inertias.extend(x)
        inertiaMatrix = create_I(x)
        if len(A) == 6*iterations:
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

# inertiaMatrix_0 = inertiaMatrix * 4
print("I =", inertiaMatrix)
for r in df.rolling(window=2):
    if len(r) < 2:
        continue
    t_1 = r['time'].values[0] / 1e6
    t_2 = r['time'].values[1] / 1e6
    dt = t_2 - t_1
    ddt = dt/2000
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


ax.set_ylabel("Angular velocity (rad/s)")
ax.set_xlabel("Time (s)")

# ax.set_ylabel("MSRE (-)")
# ax.set_xlabel("Sample size")

plt.plot(Time, X, label="X", color="tab:blue")
plt.plot(Time, Y, label="Y", color="tab:orange")
plt.plot(Time, Z, label="Z", color="tab:green")

plt.plot(Time, X_f, label=f"X ({iterations} iter.)", color="tab:blue", linestyle="dashed")
plt.plot(Time, Y_f, label=f"Y ({iterations} iter.)", color="tab:orange", linestyle="dashed")
plt.plot(Time, Z_f, label=f"Z ({iterations} iter.)", color="tab:green", linestyle="dashed")

plt.axvline([df["time"].values[iterations-1]/1e6], color="gray", linestyle="dashed")

plt.legend()
# plt.show()

error = []
for i in range(len(inertias)//6):
    e = 0
    for j in range(6):
        e += (inertias[j::6][i] / inertias[j::6][-1])**2
    error.append(e/6)

# ax = plt.figure().add_subplot(111)
# print(inertias[0::6])
# plt.plot(inertias[0::6])
# plt.plot(inertias[1::6])
# plt.plot(inertias[2::6])
# plt.plot(inertias[3::6])
# plt.plot(inertias[4::6])
# plt.plot(inertias[5::6])
# plt.plot(error)
# ax.set_yscale('log')

plt.savefig(f"sim-{iterations}.pdf", dpi=500)
plt.show()

