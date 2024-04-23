import sys
import time
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

fit_degree = 6  # Degree of the polynomial fit for

fig, ax = plt.subplots()  # Initialise plot
testfile = "test4"
df = pd.read_csv(testfile + ".csv")  # Load data file

T = df['time'].values.reshape(-1, 1) / 1e6  # Get times from the data, scaled to seconds

# Get gyroscope data and convert to rad/s
omega_x = df['gyroADC[0]'].values.reshape(-1, 1) * math.pi / 180
omega_y = df['gyroADC[1]'].values.reshape(-1, 1) * math.pi / 180
omega_z = df['gyroADC[2]'].values.reshape(-1, 1) * math.pi / 180

# Define regressed omegas in three dimensions. In addition, take the derivative to find the omega_dots
# regressedOmega_x = make_pipeline(PolynomialFeatures(degree=fit_degree), LinearRegression(fit_intercept=True))
# regressedOmega_x.fit(T, omega_x)
# regressedOmega_dot_x = np.polyder(np.flip(regressedOmega_x.named_steps.linearregression.coef_).reshape(-1))

# regressedOmega_y = make_pipeline(PolynomialFeatures(degree=fit_degree), LinearRegression(fit_intercept=True))
# regressedOmega_y.fit(T, omega_y)
# regressedOmega_dot_y = np.polyder(np.flip(regressedOmega_y.named_steps.linearregression.coef_).reshape(-1))

# regressedOmega_z = make_pipeline(PolynomialFeatures(degree=fit_degree), LinearRegression(fit_intercept=True))
# regressedOmega_z.fit(T, omega_z)
# regressedOmega_dot_z = np.polyder(np.flip(regressedOmega_z.named_steps.linearregression.coef_).reshape(-1))

# Redefine T to be uniformly distributed over the domain
T = np.arange(np.min(T), np.max(T), 0.01).reshape(-1, 1)

acceleration_x = np.array(df['accSmooth[0]'].values) * 9.81 / 2048
acceleration_y = np.array(df['accSmooth[1]'].values) * 9.81 / 2048
acceleration_z = np.array(df['accSmooth[2]'].values) * 9.81 / 2048

plotmode = 0
if plotmode == 0:
    # plt.plot(T, regressedOmega_x.predict(T), color="gray", linestyle="dotted")  # Plot fitted angular velocity
    # plt.plot(T, np.polyval(regressedOmega_dot_x, T))  # Plot angular acceleration
    plt.plot(df["time"]/1e6, df["gyroADC[0]"] * math.pi / 180, color="gray", linestyle="dotted")  # Plot measured angular velocity

    # plt.plot(T, regressedOmega_y.predict(T), color="gray", linestyle="dotted")  # Plot fitted angular velocity
    # plt.plot(T, np.polyval(regressedOmega_dot_y, T))  # Plot angular acceleration
    plt.plot(df["time"]/1e6, df["gyroADC[1]"] * math.pi / 180, color="gray", linestyle="dotted")  # Plot measured angular velocity

    # plt.plot(T, regressedOmega_z.predict(T), color="gray", linestyle="dotted")  # Plot fitted angular velocity
    # plt.plot(T, np.polyval(regressedOmega_dot_z, T))  # Plot angular acceleration
    plt.plot(df["time"]/1e6, df["gyroADC[2]"] * math.pi / 180, color="gray", linestyle="dotted")  # Plot measured angular velocity


def create_I(x):
    return np.array([[x[0], x[1], x[3]],
                     [x[1], x[2], x[4]],
                     [x[3], x[4], x[5]]])


# Arbitrarily define the locations of the coefficients.
# Note that it is always symmetric due to the definition of the product moments of inertia.

omega_0 = np.array([df["gyroADC[0]"].values[0],
                    df["gyroADC[1]"].values[0],
                    df["gyroADC[2]"].values[0]]) * math.pi / 180
zero_array = np.ones((1, 1)).reshape(-1, 1) * min(T)
# omega_0 = np.array([regressedOmega_x.predict(zero_array),
#                     regressedOmega_y.predict(zero_array),
#                     regressedOmega_z.predict(zero_array)]).reshape(-1)
# Define initial condition for the simulation later

A = []  # Define the A matrix to be used later
inertiaMatrix = None  # Define the inertia matrix
inertiaMatrix_0 = None  # Define a second inertia matrix to be compared to the end result (using all available data)
inertias = []  # Keep track of the different x vectors

iterations = 100  # Calculate the inertia tensor after 100 iterations
start_time = time.time()  # Find current time to track computation time.

rList = []
aList = []

imu_offset_A = []
imu_offset_B = []

ATA = np.zeros(36).reshape(6, 6)
prev_x = np.zeros(6)
x_errors = []


# def isTumbling(window):
#     # Check acceleration
#     tumbling = False
#     linearAcceleration = np.array([r['accSmooth[0]'].values * 9.81 / 2048,
#                                    r['accSmooth[1]'].values * 9.81 / 2048,
#                                    r['accSmooth[2]'].values * 9.81 / 2048])
#     if (np.abs(linearAcceleration) < 0.2).all():
#         tumbling = True
#
#     return tumbling


# def IMU_offset(a, o, o_dot):
#     # Matrix to find the distance between IMU and CG
#     B = []  # matrix for IMU distance
#     line_x_d = [-o[1] ** 2 - o[2] ** 2, o[0] * o[1] - o_dot[2], o[0] * o[2] + o_dot[1]]
#     line_y_d = [o[0] * o[1] + o_dot[2], -o[0] ** 2 - o[2] ** 2, o[1] * o[2] - o_dot[0]]
#     line_z_d = [o[0] * o[2] - o_dot[1], o[1] * o[2] + o_dot[0], -o[0] ** 2 - o[1] ** 2]
#     B.append(line_x_d)
#     B.append(line_y_d)
#     B.append(line_z_d)
#     imu_offset_A.append(B)
#
#     # theta = np.matrix([line_x_d, line_y_d, line_z_d])
#     # t_start = time.time()
#     # inverse_theta = np.linalg.inv(theta)
#
#     a_cg = np.zeros((3, 1))  # CG linear acceleration
#     a_difference = a_cg - a.reshape(-1, 1)
#     imu_offset_B.extend(a_difference)
#
#     # r = inverse_theta @ a_difference # r = distance between IMU & CG
#     a = np.array(imu_offset_A).reshape(-1, 3)
#     b = np.array(imu_offset_B).reshape(-1, 1)
#     r = np.linalg.inv(a.T @ a) @ a.T @ b
#
#     o_dot = o_dot.reshape(-1)
#     r = r.flatten()
#     o = o.reshape(-1)
#
#     a_cg = a.reshape(-1) + np.cross(o_dot, r) + np.cross(o, np.cross(o, r))
#     aList.append(a_cg)
#     rList.append(r)
#
#     return a_cg, r

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


m = 50  # for (m-1) order accurate estimate
coefficients = derivativeCoefficients(m)
f = 1

has_converged = False
windowSize = m * f
wasTumbling = False

times = []
derivs = []
for r in df.rolling(window=windowSize):
    if len(r) != windowSize:
        continue

    # if not wasTumbling:
    #     if isTumbling(r):
    #         pythonshutup = False
    #     else: continue

    t = r['time'].values[0] / 1e6
    h = (r['time'].values[-1] - r['time'].values[-2]) * (m+1) * f / 1e6

    times.append(t)

    linearAcceleration = np.array([r['accSmooth[0]'].values[-1] * 9.81 / 2048,
                                   r['accSmooth[1]'].values[-1] * 9.81 / 2048,
                                   r['accSmooth[2]'].values[-1] * 9.81 / 2048])

    o = np.array([r['gyroADC[0]'].values[-1],
                  r['gyroADC[1]'].values[-1],
                  r['gyroADC[2]'].values[-1]]) * math.pi / 180

    x = scipy.signal.savgol_filter(r['gyroADC[0]'].values, window_length=windowSize, polyorder=2) * math.pi / 180
    y = scipy.signal.savgol_filter(r['gyroADC[1]'].values, window_length=windowSize, polyorder=2) * math.pi / 180
    z = scipy.signal.savgol_filter(r['gyroADC[2]'].values, window_length=windowSize, polyorder=2) * math.pi / 180

    o_dot = np.array([(np.flip(x[f-1::f]) @ coefficients.reshape(-1, 1)),
                      (np.flip(y[f-1::f]) @ coefficients.reshape(-1, 1)),
                      (np.flip(z[f-1::f]) @ coefficients.reshape(-1, 1))]).reshape(-1) / (1/m*h)

    derivs.append(o_dot)

    # print(o_dot_t, o_dot, o_dot_t - o_dot)

    # Matrix to find the distance between IMU and CG
    B = []  # matrix for IMU distance
    line_x_d = [-o[1] ** 2 - o[2] ** 2, o[0] * o[1] - o_dot[2], o[0] * o[2] + o_dot[1]]
    line_y_d = [o[0] * o[1] + o_dot[2], -o[0] ** 2 - o[2] ** 2, o[1] * o[2] - o_dot[0]]
    line_z_d = [o[0] * o[2] - o_dot[1], o[1] * o[2] + o_dot[0], -o[0] ** 2 - o[1] ** 2]
    B.append(line_x_d)
    B.append(line_y_d)
    B.append(line_z_d)
    imu_offset_A.append(B)

    theta = np.matrix([line_x_d, line_y_d, line_z_d])
    t_start = time.time()
    inverse_theta = np.linalg.inv(theta)

    a_cg = np.zeros((3, 1))  # CG linear acceleration
    a_difference = a_cg - linearAcceleration.reshape(-1, 1)
    imu_offset_B.extend(a_difference)

    # r = inverse_theta @ a_difference # r = distance between IMU & CG
    a = np.array(imu_offset_A).reshape(-1, 3)
    b = np.array(imu_offset_B).reshape(-1, 1)
    r = np.linalg.inv(a.T @ a) @ a.T @ b

    o_dot = o_dot.reshape(-1)
    r = r.flatten()
    o = o.reshape(-1)

    a_cg = linearAcceleration.reshape(-1) + np.cross(o_dot, r) + np.cross(o, np.cross(o, r))
    aList.append(a_cg)
    rList.append(r)

    # The next few lines contain the massive worked-out matrix lines, rewritten to solve for I
    line_x = [o_dot[0],
              -o[2] * o[0] + o_dot[1],
              -o[2] * o[1],
              o[1] * o[0] + o_dot[2],
              o[1] ** 2 - o[2] ** 2,
              o[1] * o[2]]
    line_y = [o[2] * o[0],
              o[2] * o[1] + o_dot[0],
              o_dot[1],
              o[2] ** 2 - o[0] ** 2,
              -o[0] * o[1] + o_dot[2],
              -o[0] * o[2]]
    line_z = [-o[1] * o[0],
              o[0] ** 2 - o[1] ** 2,
              o[0] * o[1],
              -o[1] * o[2] + o_dot[0],
              o[0] * o[2] + o_dot[1],
              o_dot[2]]
    A.append(line_x)
    A.append(line_y)
    A.append(line_z)
    zeta = np.matrix([line_x, line_y, line_z]).reshape((3, 6))

    delta = np.matmul(zeta.T, zeta)
    ATA += delta

    if len(A) >= 6:  # Only proceed if the matrix isn't underdetermined
        # a = np.array(A).reshape(-1, 6)
        # b = np.zeros(len(A)).T

        # x, r, R, s = np.linalg.lstsq(a.astype('float'), b.astype('float'), rcond=-1) # We decided not to use this, since it returns the trivial solution

        # Don't worry if you're surprised that we are using eigenvectors here. I'd be offended if you weren't.
        # There's a proof for this in the final paper, but just know that we have to do something different since the
        # system that we're trying to solve is both overdetermined and homogeneous. It just turns out to be equivalent
        # to finding the eigenvector with the smallest eigenvalue of A transpose times A. Note that this way, the
        # length of the x vector is exactly 1, since that was the constraint we set to keep it from converging to
        # the trivial solution.
        eigen_values, eigen_vectors = np.linalg.eig(ATA)
        x = eigen_vectors[:, eigen_values.argmin()]

        # Assure the result is physically accurate (principal moments of inertia are positive)
        # If they cannot all be, keep the absolute value for the type 2 plot (compromise)
        if x[0] < 0:
            if not (x[2] < 0 and x[5] < 0):
                X = np.abs(x)
            else:
                X = -x
        else:
            X = x

        if len(A) != 9:
            delta = X - prev_x
            error_estimate = np.linalg.norm(delta)
            # print(np.linalg.norm(delta))
            x_errors.append(error_estimate)

            if (error_estimate <= 10e-5 and not has_converged):
                inertiaMatrix_0 = create_I(X)
                iterations = len(A) // 6
                print("Test took %s seconds to converge at %i timesteps" % (time.time() - start_time, iterations))
                has_converged = True
            prev_x = X
        else:
            prev_x = X

        inertias.extend(X)
        inertiaMatrix = create_I(X)
        # if len(A) == 6 * iterations: # After a set time, save the inertia matrix to a separate variable to compare.
        #     inertiaMatrix_0 = create_I(x)
        #     print("Test took %s seconds" % (time.time() - start_time))
if not has_converged:
    inertiaMatrix_0 = inertiaMatrix
    iterations = len(A) // 6
    print("Test took %s seconds to converge at %i" % (time.time() - start_time, iterations))
print("Full took %s seconds" % (time.time() - start_time))

# plt.imshow(inertiaMatrix, interpolation='none')
# plt.colorbar()
# plt.show()

print("I =", inertiaMatrix)  # Output calculated inertia matrix

inertia_evals, inertia_evec = np.linalg.eig(inertiaMatrix)
diag = np.linalg.inv(inertia_evec) @ inertiaMatrix @ inertia_evec
principal_transform = np.linalg.inv(inertia_evec) @ inertiaMatrix @ inertia_evec @ np.linalg.inv(inertiaMatrix)

theta_x = math.atan2(principal_transform[2, 1], principal_transform[2, 2])
theta_y = math.atan2(-principal_transform[2, 0],
                     math.sqrt(principal_transform[2, 1] ** 2 + principal_transform[2, 2] ** 2))
theta_z = math.atan2(principal_transform[1, 0], principal_transform[0, 0])
print(f"theta_x = {theta_x * 180 / math.pi: 0.2f} deg")
print(f"theta_y = {theta_y * 180 / math.pi: 0.2f} deg")
print(f"theta_z = {theta_z * 180 / math.pi: 0.2f} deg")

# === Verification by simulation ===

# Initialise simulations
omega = omega_0
angularVelocities = []

verificationOmega = omega_0
verificationAngularVelocities = []

Time = []

# Iterate through the datapoints and simulate
if 0 <= plotmode <= 2:
    for r in df.rolling(window=2):
        if len(r) < 2:
            continue  # Only run if the window is filled

        # Initialise begin and end times for interval
        t_1 = r['time'].values[0] / 1e6
        t_2 = r['time'].values[1] / 1e6
        dt = t_2 - t_1

        ddt = dt / 100  # Iterate 2000 times between datapoints

        # i = 0
        inv = np.linalg.inv(inertiaMatrix)
        inv_f = np.linalg.inv(inertiaMatrix_0)
        for t in np.arange(t_1, t_2, ddt):
            # Simulate by solving the Euler rotation equation for the angular acceleration and using it
            # to numerically integrate the angular velocity
            omega_dot = np.matmul(inv, -np.cross(omega, np.matmul(inertiaMatrix, omega)))
            # print(-np.divide(inertiaMatrix @ omega_dot * ddt, np.cross(omega, inertiaMatrix @ omega)))
            # print(inertiaMatrix @ omega_dot, np.cross(omega, inertiaMatrix @ omega))
            omega = omega + omega_dot * ddt

            omega_dot_f = np.matmul(inv_f, -np.cross(verificationOmega, np.matmul(inertiaMatrix_0, verificationOmega)))
            verificationOmega = verificationOmega + omega_dot_f * ddt

        angularVelocities.append(omega)
        verificationAngularVelocities.append(omega)
        Time.append(t_1)

    Time = np.array(Time)  # Convert time axis to numpy array


def plotVector(X, Y, labels=["x", "y", "z"], colors=["tab:blue", "tab:orange", "tab:green"], linestyle="solid"):
    Y = np.concatenate(Y)
    for i, c in enumerate(colors):
        plt.plot(X, Y[i::3], label=labels[i], color=c, linestyle=linestyle)

# Matplotlib pizazz
if plotmode == 0:
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_xlabel("Time (s)")

    # plotVector(np.array(times) + (df['time'].values[windowSize-1] - df['time'].values[0])/1e6, derivs, labels=[
    #     f"x'",
    #     f"y'",
    #     f"z'"])

    # plotVector(Time, angularVelocities)
    plotVector(Time, verificationAngularVelocities,
               labels=[f"x ({iterations} iter.)",
                       f"y ({iterations} iter.)",
                       f"z ({iterations} iter.)"])
    plt.axvline([df["time"].values[iterations - 1] / 1e6], color="gray", linestyle="dashed")

    plt.legend()
    plt.savefig(f"sim-{testfile}-{iterations}.pdf", dpi=500)
elif plotmode == 1:
    # Calculate error for convergence analysis
    error = []
    for i in range(len(inertias) // 6):
        e = 0
        for j in range(6):
            e += (inertias[j::6][i] / inertias[j::6][-1]) ** 2
        error.append(e / 6)

    ax.set_ylabel("MSRE (-)")
    ax.set_xlabel("Sample size")

    plt.plot(error)
    plt.savefig(f"error-{testfile}.pdf", dpi=500)
elif plotmode == 2:

    # ax = plt.figure().add_subplot(111)
    # print(inertias[0::6])
    plt.plot(inertias[0::6], label=r"$I_{xx}$")
    plt.plot(inertias[2::6], label=r"$I_{yy}$")
    plt.plot(inertias[5::6], label=r"$I_{zz}$")
    plt.plot(inertias[1::6], label=r"$I_{xy}$")
    plt.plot(inertias[3::6], label=r"$I_{xz}$")
    plt.plot(inertias[4::6], label=r"$I_{yz}$")
    plt.subplots_adjust(right=0.85)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    # ax.set_yscale('log')
    ax.set_ylabel("Relative inertia (-)")
    ax.set_xlabel("Sample size (-)")

    plt.savefig(f"individuals-{testfile}.pdf", bbox_inches="tight", dpi=500)
elif plotmode == 3:
    plt.plot(x_errors)
    plt.axvline([iterations], color="gray", linestyle="dashed")
    ax.set_yscale('log')
    ax.set_xlabel("Sample size (-)")

    plt.savefig(f"delta-{testfile}.pdf", bbox_inches="tight", dpi=500)
elif plotmode == 4:
    plt.figure(figsize=(10, 6))

    rList = np.concatenate(rList)
    # Plotting for x-component
    plt.subplot(3, 1, 1)
    plt.plot(rList[0::3])
    plt.title('Estimate of offset between IMU and CG')
    plt.ylabel(f'Distance in\n$x$-direction (m)')

    # Plotting for y-component
    plt.subplot(3, 1, 2)
    plt.plot(rList[1::3])
    plt.ylabel(f'Distance in\n$y$-direction (m)')

    # Plotting for z-component
    plt.subplot(3, 1, 3)
    plt.plot(rList[2::3])
    plt.xlabel('Data points (-)')
    plt.ylabel(f'Distance in\n$z$-direction (m)')

    plt.tight_layout()
    plt.savefig(f"offset-{testfile}.pdf", bbox_inches="tight", dpi=500)
elif plotmode == 5:
    plt.figure(figsize=(10, 6))

    aList = np.concatenate(aList)

    # Plotting for x-component
    plt.subplot(3, 1, 1)
    plt.plot(acceleration_x, label=r"Raw")
    plt.plot(aList[0::3], label=r'Corrected')
    plt.title('Linear acceleration')
    plt.ylabel(f'Acceleration in\n$x$-direction (m/s$^2$)')
    plt.legend()

    # Plotting for y-component
    plt.subplot(3, 1, 2)
    plt.plot(acceleration_y, label="Raw")
    plt.plot(aList[2::3], label='Corrected')
    plt.ylabel(f'Acceleration in\n$y$-direction (m/s$^2$)')
    plt.legend()

    # Plotting for z-component
    plt.subplot(3, 1, 3)
    plt.plot(acceleration_z, label="Raw")
    plt.plot(aList[2::3], label='Corrected')
    plt.xlabel('Data points')
    plt.ylabel(f'Acceleration in\n$z$-direction (m/s$^2$)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"la-{testfile}.pdf", bbox_inches="tight", dpi=500)
elif plotmode == 6:
    T = df['time'].values.reshape(-1, 1) / 1e6  # Get times from the data, scaled to seconds
    x_sp = np.abs(np.fft.fft(acceleration_x))
    y_sp = np.abs(np.fft.fft(acceleration_y))
    z_sp = np.abs(np.fft.fft(acceleration_z))
    F = np.fft.fftfreq(T.reshape(-1).shape[-1], d=T[1] - T[0])

    plt.yscale("log")

    plt.plot(F, x_sp, label=f"$x$")
    plt.plot(F, y_sp, label=f"$y$")
    plt.plot(F, z_sp, label=f"$z$")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(f"fft-{testfile}.pdf", bbox_inches="tight", dpi=500)
plt.show()
