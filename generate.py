import sys
import time
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy


# Arbitrarily define the locations of the coefficients.
# Note that it is always symmetric due to the definition of the product moments of inertia.
def create_I(x):
    return np.array([[x[0], x[1], x[3]],
                     [x[1], x[2], x[4]],
                     [x[3], x[4], x[5]]])


inertiaMatrix = create_I([
    1,
    0,
    10,
    0,
    0,
    10
])
initialOmega = np.array([
    20,
    40,
    80
])

dt = 0.001
Ndt = 1000
T = 1.

inv = np.linalg.inv(inertiaMatrix)
omega = initialOmega
angularVelocitiesX = []
angularVelocitiesY = []
angularVelocitiesZ = []
angularAccelerationsX = []
angularAccelerationsY = []
angularAccelerationsZ = []

for t in np.arange(0, T, dt):
    # Simulate by solving the Euler rotation equation for the angular acceleration and using it
    # to numerically integrate the angular velocity
    for i in range(Ndt):
        omega_dot = np.matmul(inv, -np.cross(omega, np.matmul(inertiaMatrix, omega)))
        omega = omega + omega_dot * (dt / Ndt)
    angularVelocitiesX.append(omega[0])
    angularVelocitiesY.append(omega[1])
    angularVelocitiesZ.append(omega[2])
    angularAccelerationsX.append(omega_dot[0])
    angularAccelerationsY.append(omega_dot[1])
    angularAccelerationsZ.append(omega_dot[2])

print(len(angularVelocitiesX))

zeroCol = np.zeros(round(T/dt))
df = pd.DataFrame(list(zip(
    np.arange(0, T, dt),
    angularVelocitiesX,
    angularVelocitiesY,
    angularVelocitiesZ,
    angularAccelerationsX,
    angularAccelerationsY,
    angularAccelerationsZ,
    zeroCol,
    zeroCol,
    zeroCol)), columns=["time", "gyroADC[0]", "gyroADC[1]", "gyroADC[2]", "gyroACC[0]", "gyroACC[1]", "gyroACC[2]",  "accSmooth[0]", "accSmooth[1]", "accSmooth[2]"])

df.to_csv("gen/test.csv")
