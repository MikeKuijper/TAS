import sys
import time
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy

testfile = "low_throttle"
df = pd.read_csv("test_csv/" + testfile + ".csv")  # Load data file

df['a_dot'] = pd.Series(np.zeros(len(df["time"])), index=df.index)
df['a'] = pd.Series(np.zeros(len(df["time"])), index=df.index)

T = df['time'].values.reshape(-1, 1) / 1e6  # Get times from the data, scaled to seconds

# Get gyroscope data and convert to rad/s
omega_x = df['gyroADC[0]'].values.reshape(-1, 1) * math.pi / 180
omega_y = df['gyroADC[1]'].values.reshape(-1, 1) * math.pi / 180
omega_z = df['gyroADC[2]'].values.reshape(-1, 1) * math.pi / 180

# Redefine T to be uniformly distributed over the domain
T = np.arange(np.min(T), np.max(T), 0.01).reshape(-1, 1)

acceleration_x = np.array(df['accSmooth[0]'].values) * 9.81 / 2048
acceleration_y = np.array(df['accSmooth[1]'].values) * 9.81 / 2048
acceleration_z = np.array(df['accSmooth[2]'].values) * 9.81 / 2048

plotmode = 0  # Define a 'plotmode' for deciding what to plot after the analysis.


# Arbitrarily define the locations of the coefficients.
# Note that it is always symmetric due to the definition of the product moments of inertia.
def create_I(x):
    return np.array([[x[0], x[1], x[3]],
                     [x[1], x[2], x[4]],
                     [x[3], x[4], x[5]]])


A = []  # Define the A matrix to be used later
inertiaMatrix = None  # Define the inertia matrix
inertiaMatrix_0 = None  # Define a second inertia matrix to be compared to the end result (using all available data)
inertias = []  # Keep track of the different x vectors

iterations = 100  # Calculate the inertia tensor after 100 iterations
start_time = time.time()  # Find current time to track computation time.

rList = []
aList = []

cg_matrix_A = []
cg_matrix_B = []

ATA = np.zeros(36).reshape(6, 6)
prev_x = np.zeros(6)
x_errors = []

start_threshold = 25
start_threshold_jerk = 15
stop_threshold = 15
stop_threshold_jerk = 20
min_omega = 2

betweenPeaks = False


def startsTumbling(window):
    tumbling = False

    jerk_old = window["a_dot"].values[-2]
    jerk_new = window["a_dot"].values[-1]

    omega_x = window['gyroADC[0]'].values[-1] * math.pi / 180
    omega_y = window['gyroADC[1]'].values[-1] * math.pi / 180
    omega_z = window['gyroADC[2]'].values[-1] * math.pi / 180
    omega = math.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)

    linacc_old = window["a"].values[-2]
    linacc_new = window["a"].values[-1]

    global betweenPeaks
    global startpeak
    global endpeak
    currentTime = window["time"].values[-1] / 1e6

    if linacc_new > start_threshold and linacc_old < start_threshold:
        startpeak = currentTime
    if linacc_new < start_threshold and linacc_old > start_threshold:
        endpeak = currentTime

        if 0.1 <= (endpeak - startpeak) <= 0.3 and omega > min_omega:
            betweenPeaks = True

    if betweenPeaks and abs(jerk_new) < start_threshold_jerk and not abs(jerk_old) < start_threshold_jerk:
        tumbling = True

    return tumbling


def stopsTumbling(window):
    return False # HARDCODE
    stops_tumbling = False

    jerk_old = window["a_dot"].values[-2]
    jerk_new = window["a_dot"].values[-1]

    linacc_old = window["a"].values[-2]
    linacc_new = window["a"].values[-1]

    global betweenPeaks
    if abs(jerk_new) > stop_threshold_jerk and abs(jerk_old) < stop_threshold_jerk:
        stops_tumbling = True
        betweenPeaks = False
    if linacc_new > stop_threshold and linacc_old < stop_threshold:
        stops_tumbling = True
        betweenPeaks = False

    return stops_tumbling


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


m = 30  # for (m-1) order accurate estimate
coefficients = np.flip(derivativeCoefficients(m)).reshape(-1, 1)
f = 1

has_converged = False
windowSize = m * f
wasTumbling = False

times = []
derivs = []

start_indices = []
end_indices = []

f_omegaX = []
f_omegaY = []
f_omegaZ = []

# test_ox = []
# test_oy = []
# test_oz = []
# test_ox_gt = []
# test_oy_gt = []
# test_oz_gt = []
# test_ax = []
# test_ay = []
# test_az = []
# test_t  = []

polyorder = 2
savgol_coefficients = scipy.signal.savgol_coeffs(windowSize, polyorder=polyorder, pos=windowSize - 1, use="dot")
# print(savgol_coefficients)

i = -1
for window in df.rolling(window=windowSize):
    i += 1

    w_time = window['time'] / 1e6
    w_linaccX = window['accSmooth[0]'].values * 9.81 / 2048
    w_linaccY = window['accSmooth[1]'].values * 9.81 / 2048
    w_linaccZ = window['accSmooth[2]'].values * 9.81 / 2048
    w_omegaX = window['gyroADC[0]'].values * math.pi / 180
    w_omegaY = window['gyroADC[1]'].values * math.pi / 180
    w_omegaZ = window['gyroADC[2]'].values * math.pi / 180

    memory = df[:i]

    if i % 1000 == 0:
        print(f"{i}/{len(df)}")
    if len(window) != windowSize:
        continue
    # if i + 1 == windowSize:
    f_omegaX = scipy.signal.savgol_filter(w_omegaX, window_length=windowSize, polyorder=polyorder)
    f_omegaY = scipy.signal.savgol_filter(w_omegaY, window_length=windowSize, polyorder=polyorder)
    f_omegaZ = scipy.signal.savgol_filter(w_omegaZ, window_length=windowSize, polyorder=polyorder)
    # else:
    #     x = w_angularAccelerations0.dot(savgol_coefficients)
    #     f_angularVelocitiesX = np.append(f_angularVelocitiesX, x)
    #     f_angularVelocitiesX = np.delete(angularVelocitiesX, 0)
    #     # test_ox.append(x)
    #
    #     y = w_angularAccelerations1.dot(savgol_coefficients)
    #     f_angularVelocitiesY = np.append(f_angularVelocitiesY, y)
    #     f_angularVelocitiesY = np.delete(angularVelocitiesY, 0)
    #     # test_oy.append(y)
    #
    #     z = w_angularAccelerations2.dot(savgol_coefficients)
    #     f_angularVelocitiesZ = np.append(f_angularVelocitiesZ, z)
    #     f_angularVelocitiesZ = np.delete(angularVelocitiesZ, 0)
    #     # test_oz.append(z)
    #
    #     # test_t.append(r_time.values[-1])
    #
    #     f_angularVelocitiesX_groundtruth = scipy.signal.savgol_filter(w_angularAccelerations0, window_length=windowSize, polyorder=polyorder)
    #     f_angularVelocitiesY_groundtruth = scipy.signal.savgol_filter(w_angularAccelerations1, window_length=windowSize, polyorder=polyorder)
    #     f_angularVelocitiesZ_groundtruth = scipy.signal.savgol_filter(w_angularAccelerations2, window_length=windowSize, polyorder=polyorder)
    #
    #     test_ox_gt.append(f_angularVelocitiesX_groundtruth[-1])
    #     test_oy_gt.append(f_angularVelocitiesY_groundtruth[-1])
    #     test_oz_gt.append(f_angularVelocitiesZ_groundtruth[-1])
    #
    #     angularVelocitiesX = f_angularVelocitiesX_groundtruth
    #     angularVelocitiesY = f_angularVelocitiesY_groundtruth
    #     angularVelocitiesZ = f_angularVelocitiesZ_groundtruth

    t = w_time.values[0]
    h = (w_time.values[-1] - w_time.values[-2]) * (m + 1) * f

    vec_linacc = np.array([w_linaccX[-1],
                           w_linaccY[-1],
                           w_linaccZ[-1]])

    vec_omega = np.array([w_omegaX[-1], w_omegaY[-1], w_omegaZ[-1]])

    abs_linacc = np.array([math.sqrt(w_linaccX[k] ** 2 +
                                     w_linaccY[k] ** 2 +
                                     w_linaccZ[k] ** 2) for k in range(0, windowSize)])
    f_abs_linacc = scipy.signal.savgol_filter(abs_linacc, window_length=windowSize, polyorder=2)

    vec_omega_dot = np.array([(f_omegaX[f - 1::f] @ coefficients),
                              (f_omegaY[f - 1::f] @ coefficients),
                              (f_omegaZ[f - 1::f] @ coefficients)]).reshape(-1) / (1 / m * h)
    # if i + 1 != windowSize:
    #     test_ax.append(o_dot[0])
    #     test_ay.append(o_dot[1])
    #     test_az.append(o_dot[2])

    f_abs_linacc_dot = (f_abs_linacc[f - 1::f] @ coefficients).reshape(-1) / (1 / m * h)
    df['a_dot'].values[i] = f_abs_linacc_dot[0]
    df['a'].values[i] = np.linalg.norm(f_abs_linacc[-1])

    memory['a_dot'].values[-1] = f_abs_linacc_dot[0]
    memory['a'].values[-1] = np.linalg.norm(f_abs_linacc[-1])

    derivs.append(vec_omega_dot)

    if not wasTumbling:
        if startsTumbling(memory):
            wasTumbling = True
            start_indices.append(i)
        else:
            continue
    else:
        if stopsTumbling(memory) and i - start_indices[-1] > 50:
            wasTumbling = False
            end_indices.append(i)
            continue
        # elif i - start_indices[-1] > 500:
        #     wasTumbling = False
        #     end_indices.append(i)
        #     continue

    times.append(t)

    # Matrix to find the distance between IMU and CG
    cg_submatrix_X = [-vec_omega[1] ** 2 - vec_omega[2] ** 2, vec_omega[0] * vec_omega[1] - vec_omega_dot[2], vec_omega[0] * vec_omega[2] + vec_omega_dot[1]]
    cg_submatrix_Y = [vec_omega[0] * vec_omega[1] + vec_omega_dot[2], -vec_omega[0] ** 2 - vec_omega[2] ** 2, vec_omega[1] * vec_omega[2] - vec_omega_dot[0]]
    cg_submatrix_Z = [vec_omega[0] * vec_omega[2] - vec_omega_dot[1], vec_omega[1] * vec_omega[2] + vec_omega_dot[0], -vec_omega[0] ** 2 - vec_omega[1] ** 2]
    cg_submatrix = [cg_submatrix_X, cg_submatrix_Y, cg_submatrix_Z]
    cg_matrix_A.append(cg_submatrix)

    a_cg = np.zeros((3, 1))  # CG linear acceleration, assumed zero
    a_difference = a_cg - vec_linacc.reshape(-1, 1)
    cg_matrix_B.extend(a_difference)

    a = np.array(cg_matrix_A).reshape(-1, 3)
    b = np.array(cg_matrix_B).reshape(-1, 1)
    window = np.linalg.inv(a.T @ a) @ a.T @ b
    print(window)

    vec_omega_dot = vec_omega_dot.reshape(-1)
    vec_omega = vec_omega.reshape(-1)
    window = window.flatten()

    a_cg = vec_linacc.reshape(-1) + np.cross(vec_omega_dot, window) + np.cross(vec_omega, np.cross(vec_omega, window))
    aList.append(a_cg)
    rList.append(window)

    # The next few lines contain the massive worked-out matrix lines, rewritten to solve for I
    zeta_X = [vec_omega_dot[0], -vec_omega[2] * vec_omega[0] + vec_omega_dot[1], -vec_omega[2] * vec_omega[1],
              vec_omega[1] * vec_omega[0] + vec_omega_dot[2], vec_omega[1] ** 2 - vec_omega[2] ** 2,
              vec_omega[1] * vec_omega[2]]
    A.append(zeta_X)
    zeta_Y = [vec_omega[2] * vec_omega[0], vec_omega[2] * vec_omega[1] + vec_omega_dot[0], vec_omega_dot[1],
              vec_omega[2] ** 2 - vec_omega[0] ** 2, -vec_omega[0] * vec_omega[1] + vec_omega_dot[2],
              -vec_omega[0] * vec_omega[2]]
    A.append(zeta_Y)
    zeta_Z = [-vec_omega[1] * vec_omega[0], vec_omega[0] ** 2 - vec_omega[1] ** 2, vec_omega[0] * vec_omega[1],
              -vec_omega[1] * vec_omega[2] + vec_omega_dot[0], vec_omega[0] * vec_omega[2] + vec_omega_dot[1],
              vec_omega_dot[2]]
    A.append(zeta_Z)
    zeta = np.matrix([zeta_X, zeta_Y, zeta_Z]).reshape((3, 6))

    ATA_delta = np.matmul(zeta.T, zeta)
    ATA += ATA_delta

    if len(A) >= 6:  # Only proceed if the matrix isn't under-determined
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
        inertiaCoefficients = eigen_vectors[:, eigen_values.argmin()]

        # Assure the result is physically accurate (principal moments of inertia are positive)
        # If they cannot all be, keep the absolute value for the type 2 plot (compromise)
        if inertiaCoefficients[0] < 0:
            if not (inertiaCoefficients[2] < 0 and inertiaCoefficients[5] < 0):
                X = np.abs(inertiaCoefficients)
            else:
                X = -inertiaCoefficients
        else:
            X = inertiaCoefficients

        if len(A) != 9:
            ATA_delta = X - prev_x
            error_estimate = np.linalg.norm(ATA_delta)
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

if not has_converged:
    inertiaMatrix_0 = inertiaMatrix
    iterations = len(A) // 6
    print("Test took %s seconds to converge at %i" % (time.time() - start_time, iterations))
print("Full took %s seconds" % (time.time() - start_time))
print("I =", inertiaMatrix)  # Output calculated inertia matrix


# === Verification by simulation ===

# Initialise simulations

def plotVector(plot, X, Y, labels=["x", "y", "z"], colors=["tab:blue", "tab:orange", "tab:green"], linestyle="solid"):
    Y = np.concatenate(Y)
    for i, c in enumerate(colors):
        plot.plot(X, Y[i::3], label=labels[i], color=c, linestyle=linestyle)


print(start_indices)
print(end_indices)

fig, ax = plt.subplots()  # Initialise plot

# ax.plot(df["time"] / 1e6,
#          df["a"], color="black", linestyle="dotted", alpha=0.4)  # Plot measured angular velocity
# ax.plot(df["time"] / 1e6,
#          df["a_dot"], color="tab:red", linestyle="dotted", alpha=0.4)  # Plot measured angular velocity

ax.plot(df["time"] / 1e6,
        df["gyroADC[0]"] * math.pi / 180, linestyle="dotted", color="tab:blue",
        alpha=0.8)  # Plot measured angular velocity
ax.plot(df["time"] / 1e6,
        df["gyroADC[1]"] * math.pi / 180, linestyle="dotted", color="tab:orange",
        alpha=0.8)  # Plot measured angular velocity
ax.plot(df["time"] / 1e6,
        df["gyroADC[2]"] * math.pi / 180, linestyle="dotted", color="tab:green",
        alpha=0.8)  # Plot measured angular velocity

# ax.plot(test_t, test_ox, alpha=0.8, color="tab:blue")
# ax.plot(test_t, test_oy, alpha=0.8, color="tab:orange")
# ax.plot(test_t, test_oz, alpha=0.8, color="tab:green")
#
# ax.plot(test_t, test_ax, alpha=0.5, linestyle="dashed", color="tab:blue")
# ax.plot(test_t, test_ay, alpha=0.5, linestyle="dashed", color="tab:orange")
# ax.plot(test_t, test_az, alpha=0.5, linestyle="dashed", color="tab:green")
#
# ax.plot(test_t, test_ox_gt, linestyle="dashed", color="tab:blue")
# ax.plot(test_t, test_oy_gt, linestyle="dashed", color="tab:orange")
# ax.plot(test_t, test_oz_gt, linestyle="dashed", color="tab:green")


for j in start_indices:
    plt.axvline(df["time"].values[j] / 1e6, color="gray", linestyle="dashed")
for j in end_indices:
    plt.axvline(df["time"].values[j] / 1e6, color="gray", linestyle="dashed")

plt.show()

# sys.exit()

# Iterate through the datapoints and simulate
if 0 <= plotmode <= 2:
    fig, ax = plt.subplots(3, 2)  # Initialise plot
    for i in range(len(start_indices)):
        Time = []

        omega_0 = np.array([df["gyroADC[0]"].values[start_indices[i]],
                            df["gyroADC[1]"].values[start_indices[i]],
                            df["gyroADC[2]"].values[start_indices[i]]]) * math.pi / 180
        omega = omega_0
        angularVelocities = []

        verificationOmega = omega_0
        verificationAngularVelocities = []
        end_indices.append(len(df)) # HARDCODE
        for window in df[start_indices[i]:end_indices[i]].rolling(window=2):
            if len(window) < 2:
                continue  # Only run if the window is filled

            # Initialise begin and end times for interval
            t_1 = window['time'].values[0] / 1e6
            t_2 = window['time'].values[1] / 1e6
            dt = t_2 - t_1

            ddt = dt / 100  # Iterate 2000 times between datapoints

            inv = np.linalg.inv(inertiaMatrix)
            inv_f = np.linalg.inv(inertiaMatrix_0)
            for t in np.arange(t_1, t_2, ddt):
                # Simulate by solving the Euler rotation equation for the angular acceleration and using it
                # to numerically integrate the angular velocity
                omega_dot = np.matmul(inv, -np.cross(omega, np.matmul(inertiaMatrix, omega)))
                omega = omega + omega_dot * ddt

                omega_dot_f = np.matmul(inv_f,
                                        -np.cross(verificationOmega, np.matmul(inertiaMatrix_0, verificationOmega)))
                verificationOmega = verificationOmega + omega_dot_f * ddt

            angularVelocities.append(omega)
            verificationAngularVelocities.append(omega)
            Time.append(t_1)

        Time = np.array(Time)  # Convert time axis to numpy array
        # ax = plt.subplot(3, 2, i + 1)
        ax = plt.subplot(1, 1, 1)

        if i == 2:
            ax.set_ylabel("Angular velocity (rad/s)")
        if i == 4 or i == 5:
            ax.set_xlabel("Time (s)")

        plt.title(f"Throw {i + 1}")
        plotVector(ax, Time, angularVelocities, labels=["Simulated ($x$)", "Simulated ($y$)", "Simulated ($z$)"])

        ax.plot(df[start_indices[i]:end_indices[i]]["time"] / 1e6,
                df[start_indices[i]:end_indices[i]]["gyroADC[0]"] * math.pi / 180, color="tab:blue",
                linestyle="dotted", label="Measured ($x$)")  # Plot measured angular velocity
        ax.plot(df[start_indices[i]:end_indices[i]]["time"] / 1e6,
                df[start_indices[i]:end_indices[i]]["gyroADC[1]"] * math.pi / 180, color="tab:orange",
                linestyle="dotted", label="Measured ($y$)")  # Plot measured angular velocity
        ax.plot(df[start_indices[i]:end_indices[i]]["time"] / 1e6,
                df[start_indices[i]:end_indices[i]]["gyroADC[2]"] * math.pi / 180, color="tab:green",
                linestyle="dotted", label="Measured ($z$)")  # Plot measured angular velocity

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.15, 0.5))
    plt.subplots_adjust(hspace=0.5)
    fig.set_size_inches(10, 5)
    plt.savefig(f"sim-{testfile}-final-{i}.pdf", dpi=500, bbox_inches='tight')
    plt.show()

sys.exit()

# Matplotlib pizazz
if plotmode == 0:
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_xlabel("Time (s)")

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
    plt.plot(inertias[0::6], label=r"$I_{xx}$")
    plt.plot(inertias[2::6], label=r"$I_{yy}$")
    plt.plot(inertias[5::6], label=r"$I_{zz}$")
    plt.plot(inertias[1::6], label=r"$I_{xy}$")
    plt.plot(inertias[3::6], label=r"$I_{xz}$")
    plt.plot(inertias[4::6], label=r"$I_{yz}$")
    plt.subplots_adjust(right=0.85)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
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
