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

data = UR(a,b,B,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2) + np.random.normal(size=x.size, scale=0.01)

#define objective function: returns the array to be minimized
def fcn2min(params, x, data):
    """Model a decaying sine wave and subtract data."""
    a_ = params['a_']
    b_ = params['b_']
    B_ = params['B_']
    l_0_ = params['l_0_']
    ddopller_ = params['ddopller_']
    v_LOS = params['v_LOS']
    aimag = params['aimag']
    theta = params['theta']
    Xi = params['Xi']
    eta_0 = params['eta_0']
    model = UR(a_,b_,B_,x,l_0_,ddopller_,v_LOS,aimag,theta,Xi,eta_0,J_l,J_u,L_l,L_u,S_l,S_u)
    return model - data

#Create a set of Parameters
params = Parameters()
params.add('a_',value=1,min=0,max=10)
params.add('b_',value=1,min=0,max=10)
params.add('B_',value=1,min=0,max=10000)
params.add('l_0_',value=1,min=0,max=10000)
params.add('ddopller_',value=1,min=0,max=10)
params.add('v_LOS',value=1,min=-30,max=30)
params.add('aimag',value=1,min=0,max=10)
params.add('theta',value=1,min=0,max=m.radians(180))
params.add('Xi',value=1,min=0,max=m.radians(360))
params.add('eta_0',value=1,min=0,max=30)

#do fit here with leastsq algorithm
result = minimize(fcn2min,params,args=(x,data))

#calculate final result
params_fit = result.params
final = UR(params['a_'],params['b_'],params['B_'],x,params['l_0_'],params['ddopller_'],params['v_LOS'],params['aimag'],params['theta'],params['Xi'],params['eta_0'],J_l,J_u,L_l,L_u,S_l,S_u)
report_fit(result)

#Plot results
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,data[i],'+')
    plt.plot(x,final[i])
plt.show()