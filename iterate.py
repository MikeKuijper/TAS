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


def diff(lst):
    return lst[1]-lst[0]


df = pd.read_csv("test1.csv")
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

romega_x = make_pipeline(PolynomialFeatures(degree=2), LinearRegression(fit_intercept = False))
romega_x.fit(T, omega_x)
romega_dot_x = np.polyder(romega_x.named_steps.linearregression.coef_[::-1].reshape(-1))

romega_y = make_pipeline(PolynomialFeatures(degree=2), LinearRegression(fit_intercept = False))
romega_y.fit(T, omega_y)
romega_dot_y = np.polyder(romega_y.named_steps.linearregression.coef_[::-1].reshape(-1))

romega_z = make_pipeline(PolynomialFeatures(degree=2), LinearRegression(fit_intercept = False))
romega_z.fit(T, omega_z)
romega_dot_z = np.polyder(romega_z.named_steps.linearregression.coef_[::-1].reshape(-1))

X = np.arange(min(T), max(T), 0.01).reshape(-1, 1)

plt.plot(X, romega_x.predict(X))
plt.plot(df["time"]/1e6, df["gyroADC[0]"])

plt.plot(X, romega_y.predict(X))
plt.plot(df["time"]/1e6, df["gyroADC[1]"])

plt.plot(X, romega_z.predict(X))
plt.plot(df["time"]/1e6, df["gyroADC[2]"])
plt.show()

def create_I(x):
    return np.array([[x[0], x[1], x[2]],
                     [x[1], x[3], x[4]],
                     [x[2], x[4], x[5]]])


def iterate(omega_1, omega_dot_1, omega_2, omega_dot_2):
    def it(x):
        I = create_I(x)
        r_1 = np.matmul(I, omega_dot_1) + np.cross(omega_1, np.matmul(I, omega_1))
        r_2 = np.matmul(I, omega_dot_2) + np.cross(omega_2, np.matmul(I, omega_2))
        return np.array([r_1, r_2]).reshape([-1])

    X = scipy.optimize.fsolve(it, np.ones(6))
    # print(it(X))
    return create_I(X)

for r in df.rolling(window=2):
    # if len(r) < 2 or (math.isnan(r['md_gyroADC[0]'].values[0]) or math.isnan(r['md_gyroADC[0]'].values[1])):
    if len(r) < 2:
        continue

    t_1 = r['time'].values[0]/1e6
    t_2 = r['time'].values[1]/1e6

    omega_1 = np.array([romega_x.predict(t_1.reshape(1, -1)),
                        romega_x.predict(t_1.reshape(1, -1)),
                        romega_x.predict(t_1.reshape(1, -1))]).reshape(-1) * math.pi / 180
    omega_2 = np.array([romega_x.predict(t_2.reshape(1, -1)),
                        romega_x.predict(t_2.reshape(1, -1)),
                        romega_x.predict(t_2.reshape(1, -1))]).reshape(-1) * math.pi / 180

    omega_dot_1 = np.array([np.polyval(romega_dot_x, t_1),
                            np.polyval(romega_dot_y, t_1),
                            np.polyval(romega_dot_z, t_1)]).reshape(-1) * math.pi / 180
    omega_dot_2 = np.array([np.polyval(romega_dot_x, t_2),
                            np.polyval(romega_dot_y, t_2),
                            np.polyval(romega_dot_z, t_2)]).reshape(-1) * math.pi / 180

    print(iterate(omega_1, omega_dot_1, omega_2, omega_dot_2))

# print(df["md_gyroADC[0]"])

# for i in df.index:
#     if i == 0 or i == 1:
#         continue
#
#     dt_1 = (df["time"][i-1] - df["time"][i-2]) / 1e6
#     gyro_x1 = df["gyroADC[0]"][i - 1] * math.pi / 180
#     dgx1 = (df["gyroADC[0]"][i-1] - df["gyroADC[0]"][i-2]) / dt_1 * math.pi / 180
#     gyro_y1 = df["gyroADC[1]"][i - 1] * math.pi / 180
#     dgy1 = (df["gyroADC[1]"][i-1] - df["gyroADC[1]"][i-2]) / dt_1 * math.pi / 180
#     gyro_z1 = df["gyroADC[2]"][i - 1] * math.pi / 180
#     dgz1 = (df["gyroADC[2]"][i-1] - df["gyroADC[2]"][i-2]) / dt_1 * math.pi / 180
#
#     # dt_2 = (df["time"][i] - df["time"][i-1]) / 1e6
#     # gyro_x2 = df["gyroADC[0]"][i] * math.pi / 180
#     # dgx2 = (df["gyroADC[0]"][i] - df["gyroADC[0]"][i-1]) / dt_2 * math.pi / 180
#     # gyro_y2 = df["gyroADC[1]"][i] * math.pi / 180
#     # dgy2 = (df["gyroADC[1]"][i] - df["gyroADC[1]"][i-1]) / dt_2 * math.pi / 180
#     # gyro_z2 = df["gyroADC[2]"][i] * math.pi / 180
#     # dgz2 = (df["gyroADC[2]"][i] - df["gyroADC[2]"][i-1]) / dt_2 * math.pi / 180
#
#     omega_1 = np.array([gyro_x1, gyro_y1, gyro_z1])
#     omega_dot_1 = np.array([dgx1, dgy1, dgz1])
#     omega_2 = np.array([gyro_x2, gyro_y2, gyro_z2])
#     omega_dot_2 = np.array([dgx2, dgy2, dgz2])
#     print(iterate(omega_1, omega_dot_1, omega_2, omega_dot_2))