import matplotlib.pyplot as plt
import scipy
import numpy as np

res = 0.01
X = np.arange(0, 10, res)
Y = 4*np.sin(X + 0.1) + 5 * np.sin(8 * X - 0.3) + np.random.rand(int(10/res))

butter_coefs = scipy.signal.butter(2, 5, output="ba", btype="lowpass", fs=1/res)
filtered = scipy.signal.lfilter(butter_coefs[0], butter_coefs[1], Y)

N = len(X)
yf = scipy.fft.rfft(Y)
xf = scipy.fft.rfftfreq(N, res)

plt.plot(xf, np.abs(yf))

plt.plot(X, Y, color="tab:blue", linestyle="solid")
plt.plot(X, filtered, color="tab:orange", linestyle="dashed")
plt.show()
