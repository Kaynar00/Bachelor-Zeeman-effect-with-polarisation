#%%
import numpy as np
import matplotlib.pyplot as plt
from inversion import inversion
from lmfit import Parameters, report_fit, fit_report, minimize
import math as m
from tqdm import tqdm
from Tranpol import UR


obstokes = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\observed_stokes.npy")
obwaves = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\observed_wave.npy")
cavity = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\cavity_map.npy")


# %%

plt.imshow(obstokes[0,0,:,:])
plt.xlabel('x')
plt.ylabel('y')
plt.title('I')
plt.colorbar()
plt.savefig('Istokesdata.pdf')
plt.show()

plt.imshow(obstokes[1,0,:,:])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Q')
plt.colorbar()
plt.savefig('Qstokesdata.pdf')
plt.show()

plt.imshow(obstokes[2,0,:,:])
plt.xlabel('x')
plt.ylabel('y')
plt.title('U')
plt.colorbar()
plt.savefig('Ustokesdata.pdf')
plt.show()

plt.imshow(obstokes[3,0,:,:])
plt.xlabel('x')
plt.ylabel('y')
plt.title('V')
plt.colorbar()
plt.savefig('Vstokesdata.pdf')
plt.show()

#exercise
xvalue = 400
yvalue = 400

space = 0.3

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(obwaves[:15],obstokes[i,:15,yvalue,xvalue])
plt.show()

data = obstokes[:,:15,yvalue,xvalue]

datar = data.reshape(data.shape[0]*data.shape[1])

weight = np.ones_like(data)
weight[0,:] = 2. / np.max(np.abs(data[0,:]))
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
params.add('a_',value=0.5,min=0,max=10)
params.add('b_',value=0.5,min=0,max=10)
params.add('B_',value=100,min=0,max=20000)
params.add('ddopller_',value=0.05,min=0,max=1)
params.add('v_LOS',value=0,min=-30,max=30)
params.add('aimag',value=0.2,min=0,max=1)
params.add('theta',value=m.radians(30),min=m.radians(0),max=m.radians(180))
params.add('Xi',value=m.radians(30),min=m.radians(0),max=m.radians(360))
params.add('eta_0',value=10,min=0,max=30)

result, finala = inversion(params,l_0,J_l,J_u,L_l,L_u,S_l,S_u,obwaves[:15],datar,weight)

report_fit(result)

with open("umbra.txt","w") as f:
    f.write(fit_report(result))

final = finala.reshape(data.shape[0],data.shape[1])

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(obwaves[:15],data[i],'+')
    plt.plot(obwaves[:15],final[i])
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.savefig('umbra.pdf')
plt.show()
# %%
plt.imshow(obstokes[0,0,:,:])
plt.colorbar()
plt.show()

#exercise
xvalue = 100
yvalue = 100

space = 0.3

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(obwaves[:15],obstokes[i,:15,yvalue,xvalue])
plt.show()

data = obstokes[:,:15,yvalue,xvalue]

datar = data.reshape(data.shape[0]*data.shape[1])

weight = np.ones_like(data)
weight[0,:] = 2. / np.max(np.abs(data[0,:]))
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
params.add('a_',value=0.5,min=0,max=10)
params.add('b_',value=0.5,min=0,max=10)
params.add('B_',value=100,min=0,max=20000)
params.add('ddopller_',value=0.05,min=0,max=1)
params.add('v_LOS',value=0,min=-30,max=30)
params.add('aimag',value=0.2,min=0,max=1)
params.add('theta',value=m.radians(30),min=m.radians(0),max=m.radians(180))
params.add('Xi',value=m.radians(30),min=m.radians(0),max=m.radians(360))
params.add('eta_0',value=10,min=0,max=30)

result, finala = inversion(params,l_0,J_l,J_u,L_l,L_u,S_l,S_u,obwaves[:15],datar,weight)

report_fit(result)

with open("quietsun.txt","w") as f:
    f.write(fit_report(result))

final = finala.reshape(data.shape[0],data.shape[1])

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(obwaves[:15],data[i],'+')
    plt.plot(obwaves[:15],final[i])
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.savefig('quietsun.pdf')
plt.show()
# %%

plt.imshow(obstokes[0,0,:,:])
plt.show()

#exercise
xvalue = 200
yvalue = 200

space = 0.3

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(obwaves[:15],obstokes[i,:15,yvalue,xvalue])
plt.show()

data = obstokes[:,:15,yvalue,xvalue]

datar = data.reshape(data.shape[0]*data.shape[1])

weight = np.ones_like(data)
weight[0,:] = 2. / np.max(np.abs(data[0,:]))
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
params.add('a_',value=0.5,min=0,max=10)
params.add('b_',value=0.5,min=0,max=10)
params.add('B_',value=100,min=0,max=20000)
params.add('ddopller_',value=0.05,min=0,max=1)
params.add('v_LOS',value=0,min=-30,max=30)
params.add('aimag',value=0.2,min=0,max=1)
params.add('theta',value=m.radians(30),min=m.radians(0),max=m.radians(180))
params.add('Xi',value=m.radians(30),min=m.radians(0),max=m.radians(360))
params.add('eta_0',value=10,min=0,max=30)

result,finala = inversion(params,l_0,J_l,J_u,L_l,L_u,S_l,S_u,obwaves[:15],datar,weight)

report_fit(result)

with open("penumbra.txt","w") as f:
    f.write(fit_report(result))

final = finala.reshape(data.shape[0],data.shape[1])

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(obwaves[:15],data[i],'+')
    plt.plot(obwaves[:15],final[i])
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.savefig('penumbra.pdf')
plt.show()
# %%
#exercise 2

from scipy import ndimage

skip_factor = 7
sdata = obstokes[:,:15,0::skip_factor,0::skip_factor]

result = np.empty((134,132,9))

for y in tqdm(range(134)):
    for x in range(132):
        data = sdata[:,:,y,x]
        datar = data.reshape(data.shape[0]*data.shape[1])

        weight = np.ones_like(data)
        weight[0,:] = 2. / np.max(np.abs(data[0,:]))
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
        params.add('a_',value=0.5,min=0,max=1)
        params.add('b_',value=0.5,min=0,max=1)
        params.add('B_',value=100,min=0,max=3000)
        params.add('ddopller_',value=0.05,min=0,max=0.2)
        params.add('v_LOS',value=0,min=-5,max=5)
        params.add('aimag',value=0.2,min=0,max=1)
        params.add('theta',value=m.radians(30),min=m.radians(0),max=m.radians(180))
        params.add('Xi',value=m.radians(30),min=m.radians(0),max=m.radians(180))
        params.add('eta_0',value=10,min=0,max=30)

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
            model = UR(a__,b__,B__,x_,l_0,ddopller__,v_LOS__,aimag__,theta__,Xi__,eta_0__,J_l,J_u,L_l,L_u,S_l,S_u)
            models = np.array(list(model[0])+list(model[1])+list(model[2])+list(model[3]))
            return weight_*(models - data_)

        #do fit here with leastsq algorithm
        resultr = minimize(fcn2min,params,args=(obwaves[:15],datar,weight),max_nfev=200)

        #calculate final result
        params_fit = resultr.params

        a = params_fit['a_']
        b = params_fit['b_']
        B = params_fit['B_']
        ddoppler = params_fit['ddopller_']
        v_LOS = params_fit['v_LOS']
        aimag = params_fit['aimag']
        theta = params_fit['theta']
        Xi = params_fit['Xi']
        eta_0 = params_fit['eta_0']

        result[y,x,0] = a.value
        result[y,x,1] = b.value
        result[y,x,2] = B.value
        result[y,x,3] = ddoppler.value
        result[y,x,4] = v_LOS.value
        result[y,x,5] = aimag.value
        result[y,x,6] = theta.value
        result[y,x,7] = Xi.value
        result[y,x,8] = eta_0.value

np.save('resultnosmooth.npy',result)

kernelsize = 3

result[:,:,0] = ndimage.gaussian_filter(result[:,:,0], kernelsize)

result[:,:,1] = ndimage.gaussian_filter(result[:,:,1], kernelsize)

result[:,:,2] = ndimage.gaussian_filter(result[:,:,2], kernelsize)

result[:,:,3] = ndimage.gaussian_filter(result[:,:,3], kernelsize)

result[:,:,4] = ndimage.gaussian_filter(result[:,:,4], kernelsize)

result[:,:,5] = ndimage.gaussian_filter(result[:,:,5], kernelsize)

result[:,:,6] = ndimage.gaussian_filter(result[:,:,6], kernelsize)

A = ndimage.gaussian_filter(np.sin(2*result[:,:,7]),kernelsize)

B = ndimage.gaussian_filter(np.cos(2*result[:,:,7]),kernelsize)

result[:,:,7] = (np.arctan2(A,B)% (2*np.pi) )/2.

result[:,:,8] = ndimage.gaussian_filter(result[:,:,8], kernelsize)

for y in tqdm(range(134)):
    for x in range(132):
        data = sdata[:,:,y,x]
        datar = data.reshape(data.shape[0]*data.shape[1])

        weight = np.ones_like(data)
        weight[0,:] = 2. / np.max(np.abs(data[0,:]))
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
        params.add('a_',value=result[y,x,0],min=0,max=1)
        params.add('b_',value=result[y,x,1],min=0,max=1)
        params.add('B_',value=result[y,x,2],min=0,max=3000)
        params.add('ddopller_',value=result[y,x,3],min=0,max=0.2)
        params.add('v_LOS',value=result[y,x,4],min=-5,max=5)
        params.add('aimag',value=result[y,x,5],min=0,max=1)
        params.add('theta',value=result[y,x,6],min=m.radians(0),max=m.radians(180))
        params.add('Xi',value=result[y,x,7],min=m.radians(0),max=m.radians(180))
        params.add('eta_0',value=result[y,x,8],min=0,max=30)

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
            model = UR(a__,b__,B__,x_,l_0,ddopller__,v_LOS__,aimag__,theta__,Xi__,eta_0__,J_l,J_u,L_l,L_u,S_l,S_u)
            models = np.array(list(model[0])+list(model[1])+list(model[2])+list(model[3]))
            return weight_*(models - data_)

        #do fit here with leastsq algorithm
        resultr = minimize(fcn2min,params,args=(obwaves[:15],datar,weight),max_nfev=200)

        #calculate final result
        params_fit = resultr.params

        a = params_fit['a_']
        b = params_fit['b_']
        B = params_fit['B_']
        ddoppler = params_fit['ddopller_']
        v_LOS = params_fit['v_LOS']
        aimag = params_fit['aimag']
        theta = params_fit['theta']
        Xi = params_fit['Xi']
        eta_0 = params_fit['eta_0']

        result[y,x,0] = a.value
        result[y,x,1] = b.value
        result[y,x,2] = B.value
        result[y,x,3] = ddoppler.value
        result[y,x,4] = v_LOS.value
        result[y,x,5] = aimag.value
        result[y,x,6] = theta.value
        result[y,x,7] = Xi.value
        result[y,x,8] = eta_0.value

np.save('result.npy',result)
#%%
from scipy import ndimage

result = np.load('result.npy')

plt.imshow(result[:,:,0],interpolation = None)
plt.xlabel('x')
plt.ylabel('y')
plt.title('a')
plt.colorbar()
plt.savefig('a.pdf')
plt.show()

plt.imshow(result[:,:,1],interpolation = None)
plt.xlabel('x')
plt.ylabel('y')
plt.title('b')
plt.colorbar()
plt.savefig('b.pdf')
plt.show()

plt.imshow(result[:,:,2],interpolation = None)
plt.xlabel('x')
plt.ylabel('y')
plt.title('B')
plt.colorbar()
plt.savefig('mag.pdf')
plt.show()

plt.imshow(result[:,:,3],interpolation = None)
plt.xlabel('x')
plt.ylabel('y')
plt.title('doppler width')
plt.colorbar()
plt.savefig('doppler_width.pdf')
plt.show()

plt.imshow(result[:,:,4],cmap='bwr',interpolation = None)
plt.xlabel('x')
plt.ylabel('y')
plt.title('v_LOS')
plt.colorbar()
plt.savefig('v_LOS.pdf')
plt.show()

plt.imshow(result[:,:,5],interpolation = None)
plt.xlabel('x')
plt.ylabel('y')
plt.title('aimag')
plt.colorbar()
plt.savefig('aimag.pdf')
plt.show()

plt.imshow(result[:,:,6],cmap = 'RdYlGn',interpolation = None)
plt.xlabel('x')
plt.ylabel('y')
plt.title('theta')
plt.colorbar()
plt.savefig('theta.pdf')
plt.show()

plt.imshow(result[:,:,7],cmap='twilight',interpolation = 'nearest')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Xi')
plt.colorbar()
plt.savefig('Xi.pdf')
plt.show()

plt.imshow(result[:,:,8],interpolation = None)
plt.xlabel('x')
plt.ylabel('y')
plt.title('eta_0')
plt.colorbar()
plt.savefig('eta_0.pdf')
plt.show()
# %%
