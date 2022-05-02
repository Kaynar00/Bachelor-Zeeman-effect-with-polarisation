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
        cm = cavity[0::skip_factor, 0::skip_factor]
        l0 = 6301.5 # reference waveleght
        cc = 3e5 # light speed
        result[y,x,4] = result[y,x,4] + cm[y,x]/l0*cc

np.save('result.npy',result)
# %%
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
