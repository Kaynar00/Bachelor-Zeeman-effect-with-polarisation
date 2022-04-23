#%%
import numpy as np
import matplotlib.pyplot as plt
from inversion import inversion
from lmfit import Parameters, report_fit, fit_report
import math as m


obstokes = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\observed_stokes.npy")
obwaves = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\observed_wave.npy")
cavity = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\cavity_map.npy")


# %%

plt.imshow(obstokes[0,0,:,:])
plt.show()

#exercise
xvalue = 100
yvalue = 100

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(obwaves[:15],obstokes[i,:15,yvalue,xvalue])
plt.show()

data = np.empty((4,15))

for i in range(4):
    data[i] = obstokes[i,:15,yvalue,xvalue]

datar = np.array(list(obstokes[0,:15,yvalue,xvalue])+list(obstokes[1,:15,yvalue,xvalue])+list(obstokes[2,:15,yvalue,xvalue])+list(obstokes[3,:15,yvalue,xvalue]))

weight = np.ones_like(data)
weight[0,:] = 1. / np.max(np.abs(data[0,:]))
weight[1,:] = 1. / np.max(np.abs(data[1,:]))
weight[2,:] = 1. / np.max(np.abs(data[2,:]))
weight[3,:] = 1. / np.max(np.abs(data[3,:]))
weight = weight.reshape(weight.shape[0]*weight.shape[1])

l_0 = 6301.5
J_l = 2
J_u = 2
L_l = 1
L_u = 2
S_l = 2
S_u = 2

params = Parameters()
params.add('a_',value=0,min=0,max=10)
params.add('b_',value=0,min=0,max=10)
params.add('B_',value=0,min=0,max=20000)
params.add('ddopller_',value=0,min=0,max=1)
params.add('v_LOS',value=0,min=-30,max=30)
params.add('aimag',value=0,min=0,max=1)
params.add('theta',value=0,min=m.radians(0),max=m.radians(180))
params.add('Xi',value=0,min=m.radians(0),max=m.radians(360))
params.add('eta_0',value=0,min=0,max=30)

result, finala = inversion(params,l_0,J_l,J_u,L_l,L_u,S_l,S_u,obwaves[:15],datar,weight)

report_fit(result)
# %%

