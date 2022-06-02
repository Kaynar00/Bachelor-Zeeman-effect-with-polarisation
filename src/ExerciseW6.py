#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Tranpol import UR

cavity = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\cavity_map.npy")
result = np.load('result.npy')
obstokes = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\observed_stokes.npy")
obwaves = np.load(r"C:\Users\keyur\OneDrive\Dokument\Zeemandata\observed_wave.npy")

plt.imshow(cavity)
plt.show()

plt.imshow(result[:,:,2])
plt.show()

#%%

for y in tqdm(range(134)):
    for x in range (132):
        skip_factor = 7
        cm = cavity[::skip_factor, ::skip_factor]
        l0 = 6301.5 # reference waveleght
        cc = 3e5 # light speed
        result[y,x,4] = result[y,x,4] + cm[y,x]/l0*cc

np.save('resultcavity.npy',result)
# %%
from scipy import ndimage

def make_colormap(seq):
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
import matplotlib.colors as mcolors
c = mcolors.ColorConverter().to_rgb
phimap = make_colormap([c('white'), c('tomato'), 0.33, c('tomato'), c('deepskyblue'), 0.66, c('deepskyblue'), c('white')])

result = np.load('resultcavity.npy')

name = ['$S_1$','$S_0$','$B$','$\Delta\lambda_D$','$v_{LOS}$','$a_D$',r'$\theta$',r'$\chi$','$\eta_0$']
units = ['$(1)$','$(1)$','Gauss','Ångström','km/s','$(1)$','Radians','Radians','$(1)$']

plt.subplots(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    if i == 4:
        plt.imshow(result[:,:,i],cmap='bwr',interpolation = None)
        cbar = plt.colorbar(label=units[i])
    elif i == 3:
        plt.imshow(result[:,:,i],interpolation = None,vmax=0.07)
        cbar = plt.colorbar(label=units[i])
    elif i == 6:
        plt.imshow(result[:,:,i],cmap = 'RdYlGn',interpolation = None)
        cbar = plt.colorbar(label=units[i])
    elif i == 7:
        plt.imshow(result[:,:,i],cmap=phimap,interpolation = 'nearest')
        cbar = plt.colorbar(label=units[i])
        Y, X = np.mgrid[0:result.shape[0]:result.shape[0]*1j, 0:result.shape[1]:result.shape[1]*1j]

        toadd = +45*np.pi/180. # Reference system of the azimuth towards North
        # Some smoothing or it will have artifacts in noisy areas
        A = ndimage.gaussian_filter(np.sin(2*result[:,:,i]+ toadd), 5)
        B = ndimage.gaussian_filter(np.cos(2*result[:,:,i]+ toadd), 5)
        uvmap = (np.arctan2(A,B)% (2*np.pi) )/2.


        U = np.sin(uvmap)
        V = np.cos(uvmap)

        plt.streamplot(X, Y, U, V, density=[2.0,2.0],linewidth=0.5,color ='k',arrowsize = 1e-7,maxlength=0.5)
        plt.xlim(0,result.shape[1])
        plt.ylim(result.shape[0],0)
    else:
        plt.imshow(result[:,:,i],interpolation = None)
        cbar = plt.colorbar(label=units[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(name[i])
plt.subplots_adjust(hspace = 0.3,wspace=0.6)
plt.savefig('resultfinal.pdf')
plt.show()

#%%

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
J_l = 2
J_u = 2
L_l = 1
L_u = 2
S_l = 2
S_u = 2
skip_factor = 7
sdata = obstokes[:,:15,0::skip_factor,0::skip_factor]
x = 10
y = 10
sim = UR(result[y,x,0],result[y,x,1],result[y,x,2],obwaves[0:15],6301.5,result[y,x,3],result[y,x,4],result[y,x,5],result[y,x,6],result[y,x,7],result[y,x,8],J_l,J_u,L_l,L_u,S_l,S_u)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(obwaves[:15],sdata[i,:15,y,x],'+')
    plt.plot(obwaves[:15],sim[i])
plt.savefig('quiet_sun_result')
plt.show()

x = 30
y = 30
sim = UR(result[y,x,0],result[y,x,1],result[y,x,2],obwaves[0:15],6301.5,result[y,x,3],result[y,x,4],result[y,x,5],result[y,x,6],result[y,x,7],result[y,x,8],J_l,J_u,L_l,L_u,S_l,S_u)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(obwaves[:15],sdata[i,:15,y,x],'+')
    plt.plot(obwaves[:15],sim[i])
plt.savefig('penumbra_result')
plt.show()

x = 50
y = 50
sim = UR(result[y,x,0],result[y,x,1],result[y,x,2],obwaves[0:15],6301.5,result[y,x,3],result[y,x,4],result[y,x,5],result[y,x,6],result[y,x,7],result[y,x,8],J_l,J_u,L_l,L_u,S_l,S_u)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(obwaves[:15],sdata[i,:15,y,x],'+')
    plt.plot(obwaves[:15],sim[i])
plt.savefig('umbra_result')
plt.show()
# %%
