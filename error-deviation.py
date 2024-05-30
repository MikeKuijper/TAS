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

inertiaMatrices = []

def run(name_of_file, inertiaMatrices):
    # fit_degree = 6  # Degree of the polynomial fit for

    # fig, ax = plt.subplots()  # Initialise plot
    testfile = name_of_file
    df = pd.read_csv(testfile + ".csv")  # Load data file

    # df['a_dot_x'] = pd.Series(np.zeros(len(df["time"])), index=df.index)
    # df['a_dot_y'] = pd.Series(np.zeros(len(df["time"])), index=df.index)
    # df['a_dot_z'] = pd.Series(np.zeros(len(df["time"])), index=df.index)
    df['a_dot'] = pd.Series(np.zeros(len(df["time"])), index=df.index)
    df['a'] = pd.Series(np.zeros(len(df["time"])), index=df.index)

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
    # if plotmode == 0:
        # plt.plot(T, regressedOmega_x.predict(T), color="gray", linestyle="dotted")  # Plot fitted angular velocity
        # plt.plot(T, np.polyval(regressedOmega_dot_x, T))  # Plot angular acceleration

        # plt.plot(df["time"]/1e6, df["gyroADC[0]"] * math.pi / 180, color="gray", linestyle="dotted")  # Plot measured angular velocity

        # plt.plot(T, regressedOmega_y.predict(T), color="gray", linestyle="dotted")  # Plot fitted angular velocity
        # plt.plot(T, np.polyval(regressedOmega_dot_y, T))  # Plot angular acceleration

        # plt.plot(df["time"]/1e6, df["gyroADC[1]"] * math.pi / 180, color="gray", linestyle="dotted")  # Plot measured angular velocity

        # plt.plot(T, regressedOmega_z.predict(T), color="gray", linestyle="dotted")  # Plot fitted angular velocity
        # plt.plot(T, np.polyval(regressedOmega_dot_z, T))  # Plot angular acceleration

        # plt.plot(df["time"]/1e6, df["gyroADC[2]"] * math.pi / 180, color="gray", linestyle="dotted")  # Plot measured angular velocity


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
    #     linearAcceleration = np.array([window['accSmooth[0]'].values[0] * 9.81 / 2048,
    #                                    window['accSmooth[1]'].values[0] * 9.81 / 2048,
    #                                    window['accSmooth[2]'].values[0] * 9.81 / 2048])
    #     if (np.abs(linearAcceleration) < 4).all():
    #         tumbling = True
    #
    #     return tumbling


    start_threshold = 25
    start_threshold_jerk = 15
    stop_threshold = 15
    stop_threshold_jerk = 20
    min_omega = 2
    global betweenPeaks
    betweenPeaks = False
    
    
    def startsTumbling(window):
        tumbling = False

        # jerks_old = np.array([window['a_dot_x'].values[-2],
        #                      window['a_dot_y'].values[-2],
        #                      window['a_dot_z'].values[-2]])
        # jerks_new = np.array([window['a_dot_x'].values[-1],
        #                      window['a_dot_y'].values[-1],
        #                      window['a_dot_z'].values[-1]])

        # jerk_old = np.linalg.norm(jerks_old)
        jerk_old = window["a_dot"].values[-2]
        # jerk_new = np.linalg.norm(jerks_new)
        jerk_new = window["a_dot"].values[-1]

        omega_x = window['gyroADC[0]'].values[-1] * math.pi / 180
        omega_y = window['gyroADC[1]'].values[-1] * math.pi / 180
        omega_z = window['gyroADC[2]'].values[-1] * math.pi / 180
        omega = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

        # linearAcceleration_old = np.array([window['accSmooth[0]'].values[-2],
        #                                window['accSmooth[1]'].values[-2],
        #                                window['accSmooth[2]'].values[-2]]) * 9.81 / 2048
        # linearAcceleration_new = np.array([window['accSmooth[0]'].values[-1],
        #                                window['accSmooth[1]'].values[-1],
        #                                window['accSmooth[2]'].values[-1]]) * 9.81 / 2048
        # linacc_old = np.linalg.norm(linearAcceleration_old)
        linacc_old = window["a"].values[-2]
        # linacc_new = np.linalg.norm(linearAcceleration_new)
        linacc_new = window["a"].values[-1]
        # print(linearAcceleration, 5000 * 9.81 / 2048)

        global betweenPeaks
        global startpeak
        global endpeak
        currentTime = window["time"].values[-1] / 1e6

        if linacc_new > start_threshold and linacc_old < start_threshold:
            startpeak = currentTime
        if linacc_new < start_threshold and linacc_old > start_threshold:
            endpeak = currentTime

            # print(endpeak-startpeak)
            # print(omega)
            if 0.1 <= (endpeak - startpeak) <= 0.3 and omega > min_omega:
                betweenPeaks = True

        if betweenPeaks and abs(jerk_new) < start_threshold_jerk and not abs(jerk_old) < start_threshold_jerk:
            tumbling = True

        return tumbling

    def stopsTumbling(window):
        stops_tumbling = False

        # jerks_old = np.array([window['a_dot_x'].values[-2],
        #                      window['a_dot_y'].values[-2],
        #                      window['a_dot_z'].values[-2]])
        # jerks_new = np.array([window['a_dot_x'].values[-1],
        #                      window['a_dot_y'].values[-1],
        #                      window['a_dot_z'].values[-1]])
        # jerk_old = np.linalg.norm(jerks_old)
        jerk_old = window["a_dot"].values[-2]
        # jerk_new = np.linalg.norm(jerks_new)
        jerk_new = window["a_dot"].values[-1]

        linacc_old = window["a"].values[-2]
        linacc_new = window["a"].values[-1]

        # linearAcceleration_old = np.array([window['accSmooth[0]'].values[-2],
        #                                window['accSmooth[1]'].values[-2],
        #                                window['accSmooth[2]'].values[-2]]) * 9.81 / 2048
        # linearAcceleration_new = np.array([window['accSmooth[0]'].values[-1],
        #                                window['accSmooth[1]'].values[-1],
        #                                window['accSmooth[2]'].values[-1]]) * 9.81 / 2048
        # print(linearAcceleration, 2000 * 2048 / 9.81)

        global betweenPeaks
        if abs(jerk_new) > stop_threshold_jerk and abs(jerk_old) < stop_threshold_jerk:
            stops_tumbling = True
            betweenPeaks = False
        if linacc_new > stop_threshold and linacc_old < stop_threshold:
            stops_tumbling = True
            betweenPeaks = False

        # if (np.abs(linearAcceleration_new) < stop_threshold).all() and not (np.abs(linearAcceleration_old < stop_threshold)).all():
        # if np.abs(linearAcceleration_new)

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

    #Saves the moment of inertia
    def save():
        print("I =", inertiaMatrix)
        inertiaMatrices.append(inertiaMatrix)

    #reset all used variables
    def reset():
        ATA = np.zeros(36).reshape(6, 6)
        imu_offset_A = []
        imu_offset_B = []
        inertiaMatrix = None  # Define the inertia matrix
        inertiaMatrix_0 = None  # Define a second inertia matrix to be compared to the end result (using all available data)
        inertias = []  # Keep track of the different x vectors


    m = 30  # for (m-1) order accurate estimate
    coefficients = np.flip(derivativeCoefficients(m)).reshape(-1, 1)
    f = 1

    has_converged = False
    windowSize = m * f
    wasTumbling = False

    times = []
    derivs = []
    n = 0

    # print(len(df))
    start_indices = []
    end_indices = []

    i = -1
    for r in df.rolling(window=windowSize):
        i += 1

        memory = df[:i]

        if i % 1000 == 0:
            print(f"{i}/{len(df)}")
        if len(r) != windowSize:
            continue
        elif n <= windowSize:
            n += 1
            continue

        t = r['time'].values[0] / 1e6
        h = (r['time'].values[-1] - r['time'].values[-2]) * (m + 1) * f / 1e6

        linearAcceleration = np.array([r['accSmooth[0]'].values[-1],
                                    r['accSmooth[1]'].values[-1],
                                    r['accSmooth[2]'].values[-1]]) * 9.81 / 2048

        o = np.array([r['gyroADC[0]'].values[-1],
                    r['gyroADC[1]'].values[-1],
                    r['gyroADC[2]'].values[-1]]) * math.pi / 180

        angularVelocitiesX = scipy.signal.savgol_filter(r['gyroADC[0]'].values[:n], window_length=min(n - 1, windowSize), polyorder=min(n, 2)) * math.pi / 180
        angularVelocitiesY = scipy.signal.savgol_filter(r['gyroADC[1]'].values[:n], window_length=min(n - 1, windowSize), polyorder=min(n, 2)) * math.pi / 180
        angularVelocitiesZ = scipy.signal.savgol_filter(r['gyroADC[2]'].values[:n], window_length=min(n - 1, windowSize), polyorder=min(n, 2)) * math.pi / 180

        linearAccelerations = np.array([math.sqrt(r['accSmooth[0]'].values[:n][k]**2 +
                                                r['accSmooth[1]'].values[:n][k]**2 +
                                                r['accSmooth[2]'].values[:n][k]**2) for k in range(0, n - 1)]) * 9.81 / 2048
        filteredAccelerations = scipy.signal.savgol_filter(linearAccelerations, window_length=min(n-1, windowSize),
                                                        polyorder=min(n, 2))

        accelerationsX = scipy.signal.savgol_filter(r['accSmooth[0]'].values[:n], window_length=min(n - 1, windowSize),
                                                polyorder=min(n, 2)) * 9.81 / 2048
        accelerationsY = scipy.signal.savgol_filter(r['accSmooth[1]'].values[:n], window_length=min(n - 1, windowSize),
                                                polyorder=min(n, 2)) * 9.81 / 2048
        accelerationsZ = scipy.signal.savgol_filter(r['accSmooth[2]'].values[:n], window_length=min(n - 1, windowSize),
                                                polyorder=min(n, 2)) * 9.81 / 2048

        # angularVelocitiesX = r["gyroADC[0]"].values * math.pi / 180
        # angularVelocitiesY = r["gyroADC[1]"].values * math.pi / 180
        # angularVelocitiesZ = r["gyroADC[2]"].values * math.pi / 180

        o_dot = np.array([(angularVelocitiesX[f - 1::f] @ coefficients),
                        (angularVelocitiesY[f - 1::f] @ coefficients),
                        (angularVelocitiesZ[f - 1::f] @ coefficients)]).reshape(-1) / (1 / m * h)

        # a_dot = np.array([(accelerationsX[f - 1::f] @ coefficients),
        #                   (accelerationsY[f - 1::f] @ coefficients),
        #                   (accelerationsZ[f - 1::f] @ coefficients)]).reshape(-1) / (1 / m * h)

        a_dot = (filteredAccelerations[f - 1::f] @ coefficients).reshape(-1) / (1/m*h)
        # df['a_dot_x'].values[i] = a_dot[0]
        # df['a_dot_y'].values[i] = a_dot[1]
        # df['a_dot_z'].values[i] = a_dot[2]
        # df['a_dot'].values[i] = np.linalg.norm(a_dot)
        df['a_dot'].values[i] = a_dot
        df['a'].values[i] = np.linalg.norm(filteredAccelerations[-1])

        # r['a_dot_x'].values[1] = a_dot[0]
        # r['a_dot_y'].values[1] = a_dot[1]
        # r['a_dot_z'].values[1] = a_dot[2]
        # r['a_dot'].values[1] = np.linalg.norm(a_dot)
        # r['a'].values[1] = np.linalg.norm(linearAcceleration)

        # memory['a_dot_x'].values[-1] = a_dot[0]
        # memory['a_dot_y'].values[-1] = a_dot[1]
        # memory['a_dot_z'].values[-1] = a_dot[2]
        # memory['a_dot'].values[-1] = np.linalg.norm(a_dot)
        memory['a_dot'].values[-1] = a_dot
        memory['a'].values[-1] = np.linalg.norm(filteredAccelerations[-1])

        derivs.append(o_dot)

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
                save()
                reset() 
                continue
            elif i - start_indices[-1] > 500:
                wasTumbling = False
                end_indices.append(i)
                continue


        times.append(t)

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
        # inverse_theta = np.linalg.inv(theta)

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
        line_x = [o_dot[0], -o[2] * o[0] + o_dot[1], -o[2] * o[1],
                o[1] * o[0] + o_dot[2], o[1] ** 2 - o[2] ** 2, o[1] * o[2]]
        A.append(line_x)
        line_y = [o[2] * o[0], o[2] * o[1] + o_dot[0], o_dot[1],
                o[2] ** 2 - o[0] ** 2, -o[0] * o[1] + o_dot[2], -o[0] * o[2]]
        A.append(line_y)
        line_z = [-o[1] * o[0], o[0] ** 2 - o[1] ** 2, o[0] * o[1],
                -o[1] * o[2] + o_dot[0], o[0] * o[2] + o_dot[1], o_dot[2]]
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

    def plotVector(plot, X, Y, labels=["x", "y", "z"], colors=["tab:blue", "tab:orange", "tab:green"], linestyle="solid"):
        Y = np.concatenate(Y)
        for i, c in enumerate(colors):
            plot.plot(X, Y[i::3], label=labels[i], color=c, linestyle=linestyle)

    # start_indices = [25000]
    # end_indices = [26000]

    print(start_indices)
    print(end_indices)

    fig, ax = plt.subplots()  # Initialise plot
    # ax.plot(df["time"] / 1e6,
    #          df["accSmooth[0]"] * 9.81 / 2048, color="tab:blue", linestyle="dashed")  # Plot measured angular velocity
    # ax.plot(df["time"] / 1e6,
    #          df["accSmooth[1]"] * 9.81 / 2048, color="tab:orange", linestyle="dashed")  # Plot measured angular velocity
    # ax.plot(df["time"] / 1e6,
    #          df["accSmooth[2]"] * 9.81 / 2048, color="tab:green", linestyle="dashed")  # Plot measured angular velocity
    ax.plot(df["time"] / 1e6,
            df["a"], color="black", linestyle="dotted", alpha=0.4)  # Plot measured angular velocity

    # ax.plot(df["time"] / 1e6,
    #          df["a_dot_x"], color="tab:blue", linestyle="dotted", alpha=0.4)  # Plot measured angular velocity
    # ax.plot(df["time"] / 1e6,
    #          df["a_dot_y"], color="tab:orange", linestyle="dotted", alpha=0.4)  # Plot measured angular velocity
    # ax.plot(df["time"] / 1e6,
    #          df["a_dot_z"], color="tab:green", linestyle="dotted", alpha=0.4)  # Plot measured angular velocity
    ax.plot(df["time"] / 1e6,
            df["a_dot"], color="tab:red", linestyle="dotted", alpha=0.4)  # Plot measured angular velocity


    ax.plot(df["time"] / 1e6,
            df["gyroADC[0]"] * math.pi / 180, color="tab:blue", alpha=0.2)  # Plot measured angular velocity
    ax.plot(df["time"] / 1e6,
            df["gyroADC[1]"] * math.pi / 180, color="tab:orange", alpha=0.2)  # Plot measured angular velocity
    ax.plot(df["time"] / 1e6,
            df["gyroADC[2]"] * math.pi / 180, color="tab:green", alpha=0.2)  # Plot measured angular velocity


    for j in start_indices:
        plt.axvline(df["time"].values[j] / 1e6, color="gray", linestyle="dashed")
    for j in end_indices:
        plt.axvline(df["time"].values[j] / 1e6, color="gray", linestyle="dashed")

    plt.show()

    # sys.exit()

    # Iterate through the datapoints and simulate
    if 0 <= plotmode <= 2:
        for i in range(len(start_indices)):
            # if end_indices[i] - start_indices[i] < 5:
            #     continue

            Time = []

            omega_0 = np.array([df["gyroADC[0]"].values[start_indices[i]],
                                df["gyroADC[1]"].values[start_indices[i]],
                                df["gyroADC[2]"].values[start_indices[i]]]) * math.pi / 180
            omega = omega_0
            angularVelocities = []

            verificationOmega = omega_0
            verificationAngularVelocities = []
            for r in df[start_indices[i]:end_indices[i]].rolling(window=2):
                if len(r) < 2:
                    continue  # Only run if the window is filled

                # Initialise begin and end times for interval
                t_1 = r['time'].values[0] / 1e6
                t_2 = r['time'].values[1] / 1e6
                dt = t_2 - t_1

                ddt = dt / 100  # Iterate 2000 times between datapoints

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

            fig, ax = plt.subplots()  # Initialise plot

            ax.set_ylabel("Angular velocity (rad/s)")
            ax.set_xlabel("Time (s)")

            # plotVector(ax, np.array(times) + (df['time'].values[windowSize-1] - df['time'].values[0])/1e6, derivs, labels=[
            #     f"x'",
            #     f"y'",
            #     f"z'"])

            plotVector(ax, Time, angularVelocities)
            # plotVector(ax, Time, verificationAngularVelocities,
            #            labels=[f"x ({iterations} iter.)",
            #                    f"y ({iterations} iter.)",
            #                    f"z ({iterations} iter.)"])
            # plt.axvline([df["time"].values[iterations - 1] / 1e6], color="gray", linestyle="dashed")

            ax.plot(df[start_indices[i]:end_indices[i]]["time"] / 1e6,
                    df[start_indices[i]:end_indices[i]]["gyroADC[0]"] * math.pi / 180, color="gray",
                    linestyle="dotted")  # Plot measured angular velocity
            ax.plot(df[start_indices[i]:end_indices[i]]["time"] / 1e6,
                    df[start_indices[i]:end_indices[i]]["gyroADC[1]"] * math.pi / 180, color="gray",
                    linestyle="dotted")  # Plot measured angular velocity
            ax.plot(df[start_indices[i]:end_indices[i]]["time"] / 1e6,
                    df[start_indices[i]:end_indices[i]]["gyroADC[2]"] * math.pi / 180, color="gray",
                    linestyle="dotted")  # Plot measured angular velocity

            plt.legend()
            # plt.savefig(f"sim-{testfile}-{iterations}.pdf", dpi=500)
            plt.show()
        
    return(inertiaMatrices)


def calculate_statistics(matrices):
    # Convert the list of matrices into a numpy array
    matrices_array = np.array(matrices)
    
    # Calculate the mean (average) along the first axis (axis=0)
    mean_matrix = np.mean(matrices_array, axis=0)
    
    # Calculate the standard deviation along the first axis (axis=0)
    std_matrix = np.std(matrices_array, axis=0)
    
    return mean_matrix, std_matrix

# Get Matrices
names = ['ThrowAndFlight1','ThrowAndFlight2','ThrowAndFlight3','ThrowsLab']
for name in names:
    run(name, inertiaMatrices)


# Calculate mean and standard deviation
mean_result, std_result = calculate_statistics(inertiaMatrices)

# Display the results
print("Mean (Average) Matrix:")
print(mean_result)
print("\nStandard Deviation Matrix:")
print(std_result)
