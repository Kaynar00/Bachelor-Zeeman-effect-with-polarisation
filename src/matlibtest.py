import matplotlib.pyplot as plt
import numpy as np

xarray = np.linspace(0, np.pi, 100)
yarray = [np.sin(x) for x in xarray]

plt.plot(xarray, yarray)
plt.xlabel('$\lambda - \lambda_0$')
plt.show()