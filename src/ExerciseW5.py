#%%
import numpy as np
import matplotlib.pyplot as plt


obstokes = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\observed_stokes.npy")
obwaves = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\observed_wave.npy")
cavity = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\cavity_map.npy")


# %%
#exercise
xvalue = 300
yvalue = 300

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(obstokes[i,:,yvalue,xvalue])
plt.show()
# %%
