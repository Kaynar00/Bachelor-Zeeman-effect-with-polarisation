#%%
import numpy as np
from lmfit import minimize, Parameters, report_fit
import math as m
from Tranpol import UR
import matplotlib.pyplot as plt

#Create data to be fitted

x = np.arange(6301.5-1,6301.5+1,0.04)
l_0 = 6301.5
ddopller = 0.1
B = 1000
a = 0.8
b = 0.2
J_l = 2
J_u = 2
L_l = 1
L_u = 2
S_l = 2
S_u = 2
noise = 1

data = UR(a,b,B,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2) + np.array([np.random.normal(size=x.size, scale=noise),np.random.normal(size=x.size, scale=noise),np.random.normal(size=x.size, scale=noise),np.random.normal(size=x.size, scale=noise)])
#print(len(data))
datanorm = UR(a,b,B,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)
#print(type(datanorm))
#print(type(np.random.normal(size=x.size, scale=0.001)))

#%%
print(len(data[0]))

#%%

#define objective function: returns the array to be minimized
def fcn2min(params, x, data):
    """Model a decaying sine wave and subtract data."""
    a_ = params['a_']
    b_ = params['b_']
    B_ = params['B_']
    ddopller_ = params['ddopller_']
    v_LOS = params['v_LOS']
    aimag = params['aimag']
    theta = params['theta']
    Xi = params['Xi']
    eta_0 = params['eta_0']
    model = UR(a_,b_,B_,x,l_0,ddopller_,v_LOS,aimag,theta,Xi,eta_0,J_l,J_u,L_l,L_u,S_l,S_u)
    return model - data

#Create a set of Parameters
params = Parameters()
params.add('a_',value=0.8,min=0.7,max=0.9)
params.add('b_',value=0.2,min=0.1,max=0.3)
params.add('B_',value=1000,min=900,max=2000)
params.add('ddopller_',value=0.5,min=0,max=1)
params.add('v_LOS',value=5,min=-30,max=30)
params.add('aimag',value=0.5,min=0,max=1)
params.add('theta',value=m.radians(45),min=m.radians(40),max=m.radians(50))
params.add('Xi',value=m.radians(30),min=m.radians(25),max=m.radians(35))
params.add('eta_0',value=5,min=0,max=30)

#do fit here with leastsq algorithm
result = minimize(fcn2min,params,args=(x,data))

#calculate final result
params_fit = result.params
final = UR(params_fit['a_'],params_fit['b_'],params_fit['B_'],x,l_0,params_fit['ddopller_'],params_fit['v_LOS'],params_fit['aimag'],params_fit['theta'],params_fit['Xi'],params_fit['eta_0'],J_l,J_u,L_l,L_u,S_l,S_u)
report_fit(result)

#Plot results
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,data[i],'+')
    plt.plot(x,datanorm[i])
    plt.plot(x,final[i])
plt.show()