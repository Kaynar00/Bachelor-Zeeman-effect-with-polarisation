#%%
from inversion import inversion
from lmfit import Parameters, report_fit
from Tranpol import UR
import numpy as np
import math as m
import matplotlib.pyplot as plt

#Create data to be fitted

x = np.arange(6301.5-1,6301.5+1,0.04)
l_0 = 6301.5
ddopller = 0.1
B = 1000
a = 0.8
b = 0.2
v_LOS = 10
aimag = 0.05
theta = m.radians(45)
Xi = m.radians(30)
eta_0 = 10
J_l = 2
J_u = 2
L_l = 1
L_u = 2
S_l = 2
S_u = 2
noise = 1e-5

data = UR(a,b,B,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2) + np.array([np.random.normal(size=x.size, scale=noise),np.random.normal(size=x.size, scale=noise),np.random.normal(size=x.size, scale=noise),np.random.normal(size=x.size, scale=noise)])
#print(len(data))
#datanorm = UR(a,b,B,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)
#print(type(datanorm))
#print(type(np.random.normal(size=x.size, scale=0.001)))

#print(len(data[0]))

datar = data.reshape(data.shape[0]*data.shape[1])

#Create a set of Parameters
params = Parameters()
params.add('a_',value=a,min=0.1,max=1)
params.add('b_',value=b,min=0.1,max=1)
params.add('B_',value=B,min=10,max=20000)
params.add('ddopller_',value=ddopller,min=0,max=1)
params.add('v_LOS',value=v_LOS,min=-30,max=30)
params.add('aimag',value=aimag,min=0,max=1)
params.add('theta',value=theta,min=m.radians(0),max=m.radians(180))
params.add('Xi',value=Xi,min=m.radians(0),max=m.radians(360))
params.add('eta_0',value=eta_0,min=0,max=30)

#Weights
weight = np.ones_like(data)
weight[0,:] = 1. / np.max(np.abs(data[0,:]))
weight[1,:] = 1. / np.max(np.abs(data[1,:]))
weight[2,:] = 1. / np.max(np.abs(data[2,:]))
weight[3,:] = 1. / np.max(np.abs(data[3,:]))
weight = weight.reshape(weight.shape[0]*weight.shape[1])

result,final = inversion(params,l_0,J_l,J_u,L_l,L_u,S_l,S_u,x,datar,weight)

report_fit(result)

#Plot results
finalr = final.reshape(data.shape[0],data.shape[1])
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,data[i],'+',label='Data')
    #plt.plot(x,datanorm[i],label='Data without noise')
    plt.plot(x,finalr[i],label='The fit')
    plt.legend(prop = {'size':4})
plt.savefig('Spectrafit.pdf')
plt.show()

#%%
from inversion import inversion
from lmfit import Parameters, report_fit
from Tranpol import UR
import numpy as np
import math as m
import matplotlib.pyplot as plt

#Create data to be fitted

x = np.arange(6301.5-1,6301.5+1,0.04)
l_0 = 6301.5
ddopller = 0.1
B = 100
a = 0.8
b = 0.2
v_LOS = 10
aimag = 0.05
theta = m.radians(45)
Xi = m.radians(30)
eta_0 = 10
J_l = 2
J_u = 2
L_l = 1
L_u = 2
S_l = 2
S_u = 2
noise = 0.001

data = UR(a,b,B,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2) + np.array([np.random.normal(size=x.size, scale=noise),np.random.normal(size=x.size, scale=noise),np.random.normal(size=x.size, scale=noise),np.random.normal(size=x.size, scale=noise)])
#print(len(data))
#datanorm = UR(a,b,B,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)
#print(type(datanorm))
#print(type(np.random.normal(size=x.size, scale=0.001)))

#print(len(data[0]))

datar = data.reshape(data.shape[0]*data.shape[1])

#Create a set of Parameters
params = Parameters()
params.add('a_',value=0.5,min=0.1,max=1)
params.add('b_',value=0.5,min=0.1,max=1)
params.add('B_',value=1000,min=10,max=20000)
params.add('ddopller_',value=ddopller,min=0,max=1)
params.add('v_LOS',value=v_LOS,min=-30,max=30)
params.add('aimag',value=aimag,min=0,max=1)
params.add('theta',value=theta,min=m.radians(0),max=m.radians(180))
params.add('Xi',value=Xi,min=m.radians(0),max=m.radians(360))
params.add('eta_0',value=eta_0,min=0,max=30)

#Weights
weight = np.ones_like(data)
weight[0,:] = 1. / np.max(np.abs(data[0,:]))
weight[1,:] = 1. / np.max(np.abs(data[1,:]))
weight[2,:] = 1. / np.max(np.abs(data[2,:]))
weight[3,:] = 1. / np.max(np.abs(data[3,:]))
weight = weight.reshape(weight.shape[0]*weight.shape[1])

result,final = inversion(params,l_0,J_l,J_u,L_l,L_u,S_l,S_u,x,datar,weight)

report_fit(result)

#Plot results
finalr = final.reshape(data.shape[0],data.shape[1])
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,data[i],'+',label='Data')
    #plt.plot(x,datanorm[i],label='Data without noise')
    plt.plot(x,finalr[i],label='The fit')
    plt.legend(prop = {'size':4})
plt.savefig('Spectrafit2.pdf')
plt.show()

# %%
