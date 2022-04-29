import numpy as np
import matplotlib.pyplot as plt

cavity = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\cavity_map.npy")

plt.imshow(cavity)
plt.show()

skip_factor = 7
cm = cavity[0::skip_factor, 0::skip_factor]
l0 = 6301.5 # reference waveleght
cc = 3e5 # light speed
vlos_inversion = vlos_inversion + cm/l0*cc