import numpy as np
import matplotlib.pyplot as plt
from Tranpol import UR_test,UR
import math as m

Stokes = np.load('src\stokes_milne.npy')
Stokeslabel = ['I','Q','U','V']

x = np.arange(6301.5-1,6301.5+1,0.04)
l_0 = 6301.5
ddopller = 0.1
B = 1000
a = 0.8
b = 0.2

I,Q,U,V = UR(a,b,B,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)

URlst = [I,Q,U,V]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,Stokes[i],label=Stokeslabel[i])
    plt.plot(x,URlst[i],label='Keyurs '+Stokeslabel[i])
    plt.legend()
plt.savefig('comparison_pdf')
plt.show()
