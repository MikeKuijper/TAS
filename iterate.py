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
testfile = "test3.csv"
df = pd.read_csv(testfile) # Load data file

T = df['time'].values.reshape(-1, 1) / 1e6 # Get times from the data, scaled to seconds

# Get gyroscope data and convert to rad/s
omega_x = df['gyroADC[0]'].values.reshape(-1, 1) * math.pi / 180
omega_y = df['gyroADC[1]'].values.reshape(-1, 1) * math.pi / 180
omega_z = df['gyroADC[2]'].values.reshape(-1, 1) * math.pi / 180

# Define regressed omegas in three dimensions. In addition, take the derivative to find the omega_dots
romega_x = make_pipeline(PolynomialFeatures(degree=fit_degree), LinearRegression(fit_intercept = True))
romega_x.fit(T, omega_x)
romega_dot_x = np.polyder(np.flip(romega_x.named_steps.linearregression.coef_).reshape(-1))

romega_y = make_pipeline(PolynomialFeatures(degree=fit_degree), LinearRegression(fit_intercept = True))
romega_y.fit(T, omega_y)
romega_dot_y = np.polyder(np.flip(romega_y.named_steps.linearregression.coef_).reshape(-1))

romega_z = make_pipeline(PolynomialFeatures(degree=fit_degree), LinearRegression(fit_intercept = True))
romega_z.fit(T, omega_z)
romega_dot_z = np.polyder(np.flip(romega_z.named_steps.linearregression.coef_).reshape(-1))

# Redefine T to be uniformly distributed over the domain
T = np.arange(np.min(T), np.max(T), 0.01).reshape(-1, 1)


plotmode = 0
if plotmode == 0:
    plt.plot(T, romega_x.predict(T), color="gray", linestyle="dotted") # Plot fitted angular velocity
    # plt.plot(T, np.polyval(romega_dot_x, X))  # Plot angular acceleration
    # plt.plot(df["time"]/1e6, df["gyroADC[0]"] * math.pi / 180, color="gray", linestyle="dashed")  # Plot measured angular velocity

    plt.plot(T, romega_y.predict(T), color="gray", linestyle="dotted") # Plot fitted angular velocity
    # plt.plot(T, np.polyval(romega_dot_y, X))  # Plot angular acceleration
    # plt.plot(df["time"]/1e6, df["gyroADC[1]"] * math.pi / 180, color="gray", linestyle="dashed")  # Plot measured angular velocity

    plt.plot(T, romega_z.predict(T), color="gray", linestyle="dotted") # Plot fitted angular velocity
    # plt.plot(T, np.polyval(romega_dot_z, X))  # Plot angular acceleration
    # plt.plot(df["time"]/1e6, df["gyroADC[2]"] * math.pi / 180, color="gray", linestyle="dashed")  # Plot measured angular velocity

def create_I(x):
    return np.array([[x[0], x[1], x[3]],
                     [x[1], x[2], x[4]],
                     [x[3], x[4], x[5]]])
# Arbitrarily define the locations of the coefficients.
# Note that it is always symmetric due to the definition of the product moments of inertia.

omega_0 = np.array([df["gyroADC[0]"].values[0],
                    df["gyroADC[1]"].values[0],
                    df["gyroADC[2]"].values[0]]) * math.pi / 180
# Define initial condition for the simulation later

A = []  # Define the A matrix to be used later
inertiaMatrix = None  # Define the inertia matrix
inertiaMatrix_0 = None  # Define a second inertia matrix to be compared to the end result (using all available data)
inertias = []  # Keep track of the different x vectors

iterations = 100  # Calculate the inertia tensor after 100 iterations

start_time = time.time()  # Find current time to track computation time.

ATA = np.zeros(36).reshape(6, 6)
prev_x = np.zeros(6)
x_errors = []
has_converged = False
for r in df.rolling(window=1):
    t = r['time'].values[0] / 1e6

    o = np.array([romega_x.predict(t.reshape(1, -1)),
                  romega_y.predict(t.reshape(1, -1)),
                  romega_z.predict(t.reshape(1, -1))]).reshape(-1) # Define omega (angular velocity vector)
    o_dot = np.array([np.polyval(romega_dot_x, t),
                      np.polyval(romega_dot_y, t),
                      np.polyval(romega_dot_z, t)]).reshape(-1)  # Define omega dot (angular acceleration vector)

    # The next few lines contain the massive worked-out matrix lines, rewritten to solve for I
    line_x = [o_dot[0], -o[2] * o[0] + o_dot[1], -o[2] * o[1], o[1] * o[0] + o_dot[2], o[1] ** 2 - o[2] ** 2, o[1] * o[2]]
    line_y = [o[2] * o[0], o[2] * o[1] + o_dot[0], o_dot[1], o[2] ** 2 - o[0] ** 2, -o[0] * o[1] + o_dot[2], -o[0] * o[2]]
    line_z = [-o[1] * o[0], o[0] ** 2 - o[1] ** 2, o[0] * o[1], -o[1] * o[2] + o_dot[0], o[0] * o[2] + o_dot[1], o_dot[2]]
    A.append(line_x)
    A.append(line_y)
    A.append(line_z)
    zeta = np.matrix([line_x, line_y, line_z]).reshape((3,6))
    # print(zeta)

    delta = np.matmul(zeta.T, zeta)

    # delta_norm = delta / math.sqrt(np.matmul(delta.reshape(-1), delta.reshape(-1).T))
    # ATA_norm = ATA / math.sqrt(np.matmul(ATA.reshape(-1), ATA.reshape(-1).T))
    # print("Current normalised    \n", ATA_norm)
    # print("Delta normalised      \n", delta_norm)
    # print("Normalised difference \n", ATA_norm - delta_norm)

    ATA += delta

    if len(A) >= 6: # Only proceed if the matrix isn't underdetermined
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
                iterations = len(A)//6
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
# print(inertia_ev)
diag = np.linalg.inv(inertia_evec) @ inertiaMatrix @ inertia_evec
# inertia_ev = np.linalg.inv(inertia_ev)
principal_transform = np.linalg.inv(inertia_evec) @ inertiaMatrix @ inertia_evec @ np.linalg.inv(inertiaMatrix)
# print(principal_transform)
# print(principal_transform @ inertiaMatrix)
theta_x = math.atan2(principal_transform[2, 1], principal_transform[2, 2])
theta_y = math.atan2(-principal_transform[2, 0], math.sqrt(principal_transform[2, 1] ** 2 + principal_transform[2, 2] ** 2))
theta_z = math.atan2(principal_transform[1, 0], principal_transform[0, 0])
print(f"theta_x = {theta_x * 180/math.pi: 0.2f} deg")
print(f"theta_y = {theta_y * 180/math.pi: 0.2f} deg")
print(f"theta_z = {theta_z * 180/math.pi: 0.2f} deg")

# === Verification by simulation ===

# Initialise simulations
omega = omega_0
X = []
Y = []
Z = []

omega_f = omega_0
X_f = []
Y_f = []
Z_f = []

Time = []

# Iterate through the datapoints and simulate
if 0 <= plotmode <= 2:
    for r in df.rolling(window=2):
        if len(r) < 2:
            continue # Only run if the window is filled

        # Initialise begin and end times for interval
        t_1 = r['time'].values[0] / 1e6
        t_2 = r['time'].values[1] / 1e6
        dt = t_2 - t_1

        ddt = dt/2000  # Iterate 2000 times between datapoints

        # i = 0
        inv = np.linalg.inv(inertiaMatrix)
        inv_f = np.linalg.inv(inertiaMatrix_0)
        for t in np.arange(t_1, t_2, ddt):
            # Simulate by solving the Euler rotation equation for the angular acceleration and using it
            # to numerically integrate the angular velocity
            omega_dot = np.matmul(inv, -np.cross(omega, np.matmul(inertiaMatrix, omega)))
            omega = omega + omega_dot * ddt

            omega_dot_f = np.matmul(inv_f, -np.cross(omega_f, np.matmul(inertiaMatrix_0, omega_f)))
            omega_f = omega_f + omega_dot_f * ddt

        X.append(omega[0])
        Y.append(omega[1])
        Z.append(omega[2])

        X_f.append(omega_f[0])
        Y_f.append(omega_f[1])
        Z_f.append(omega_f[2])

        Time.append(t_1)

    Time = np.array(Time)  # Convert time axis to numpy array



# Matplotlib pizazz
if plotmode == 0:
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_xlabel("Time (s)")

    plt.plot(Time, X, label="X", color="tab:blue")
    plt.plot(Time, Y, label="Y", color="tab:orange")
    plt.plot(Time, Z, label="Z", color="tab:green")

    plt.plot(Time, X_f, label=f"X ({iterations} iter.)", color="tab:blue", linestyle="dashed")
    plt.plot(Time, Y_f, label=f"Y ({iterations} iter.)", color="tab:orange", linestyle="dashed")
    plt.plot(Time, Z_f, label=f"Z ({iterations} iter.)", color="tab:green", linestyle="dashed")

    plt.axvline([df["time"].values[iterations-1]/1e6], color="gray", linestyle="dashed")

    plt.legend()
    plt.savefig(f"sim-{testfile}-{iterations}.pdf", dpi=500)
elif plotmode == 1:
    # Calculate error for convergence analysis
    error = []
    for i in range(len(inertias)//6):
        e = 0
        for j in range(6):
            e += (inertias[j::6][i] / inertias[j    ::6][-1])**2
        error.append(e/6)

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

plt.show()

