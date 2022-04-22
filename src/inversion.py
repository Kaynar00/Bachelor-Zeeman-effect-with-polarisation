#%%
import numpy as np
from lmfit import minimize, Parameters, report_fit
import math as m
from Tranpol import UR
import matplotlib.pyplot as plt

def inversion(params_,l_0_,J_l,J_u,L_l,L_u,S_l,S_u,x_,data_,weight_):
    """
    params_: Class Parameters with all the initial starting values and their limits
    l_0: rest wavelength
    J_l: the rotational quantum nuber for lower state
    J_u: the rotational quantumb number for upper state
    L_l: the angular momentum quantum number for lower state
    L_u: the angular momentum quantum number for upper state
    S_l: the spin quantum number for lower state
    S_u: the spin quantum number for upper state
    x_: All the wavelengths that where measured
    data_: Measured data
    method: Minimization method (Different methods can be found at https://lmfit.github.io/lmfit-py/fitting.html under \"Choosing Different Fitting Methods\" and should be inputted as a string)
    Minimizes the Unno-Rachkovsky model so that its as similar as possible to the data and then spots out the minimized parameters
    """
    def fcn2min(params, x_, data_,weight_):
        """Model a decaying sine wave and subtract data."""
        a__ = params['a_']
        b__ = params['b_']
        B__ = params['B_']
        ddopller__ = params['ddopller_']
        v_LOS__ = params['v_LOS']
        aimag__ = params['aimag']
        theta__ = params['theta']
        Xi__ = params['Xi']
        eta_0__ = params['eta_0']
        model = UR(a__,b__,B__,x_,l_0_,ddopller__,v_LOS__,aimag__,theta__,Xi__,eta_0__,J_l,J_u,L_l,L_u,S_l,S_u)
        models = np.array(list(model[0])+list(model[1])+list(model[2])+list(model[3]))
        return weight_*(models - data_)

    #do fit here with leastsq algorithm
    result = minimize(fcn2min,params_,args=(x_,data_,weight_))

    #calculate final result
    params_fit = result.params
    final = UR(params_fit['a_'],params_fit['b_'],params_fit['B_'],x_,l_0_,params_fit['ddopller_'],params_fit['v_LOS'],params_fit['aimag'],params_fit['theta'],params_fit['Xi'],params_fit['eta_0'],J_l,J_u,L_l,L_u,S_l,S_u)
    finala = np.array(list(final[0])+list(final[1])+list(final[2])+list(final[3]))

    return result, finala

if __name__ == '__main__':
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
    noise = 0.01

    data = UR(a,b,B,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2) + np.array([np.random.normal(size=x.size, scale=noise),np.random.normal(size=x.size, scale=noise),np.random.normal(size=x.size, scale=noise),np.random.normal(size=x.size, scale=noise)])
    #print(len(data))
    #datanorm = UR(a,b,B,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)
    #print(type(datanorm))
    #print(type(np.random.normal(size=x.size, scale=0.001)))

    #print(len(data[0]))

    datar = data.reshape(data.shape[0]*data.shape[1])

    #Create a set of Parameters
    params = Parameters()
    params.add('a_',value=a,min=0.7,max=0.9)
    params.add('b_',value=b,min=0.1,max=0.3)
    params.add('B_',value=B,min=900,max=2000)
    params.add('ddopller_',value=ddopller,min=0,max=1)
    params.add('v_LOS',value=v_LOS,min=-30,max=30)
    params.add('aimag',value=aimag,min=0,max=1)
    params.add('theta',value=theta,min=m.radians(40),max=m.radians(50))
    params.add('Xi',value=Xi,min=m.radians(25),max=m.radians(35))
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
# %%
