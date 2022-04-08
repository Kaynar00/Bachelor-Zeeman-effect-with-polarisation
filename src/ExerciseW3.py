from Tranpol import Zeemansplittest, UR
import numpy as np
import matplotlib.pyplot as plt
import math as m

#Exercise 1

Zeemansplittest(2,2,1,2,2,2)

#Exercise 2

Stokes = np.load('src\stokes_milne.npy')
Stokeslabel = ['I','Q','U','V']

x = np.arange(6301.5-1,6301.5+1,0.04)
l_0 = 6301.5
ddopller = 0.1
B = 1000
a = 0.8
b = 0.2

I2,Q2,U2,V2 = UR(a,b,B,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)

URlst = [I2,Q2,U2,V2]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,Stokes[i],label=Stokeslabel[i])
    plt.plot(x,URlst[i],label='Keyurs '+Stokeslabel[i])
    plt.legend()
plt.title('Exercise 2')
plt.savefig('ExerciseW3_2.pdf')
plt.show()

#Exercise 3

I3,Q3,U3,V3 = UR(a,b,10,x,l_0,ddopller,0,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)

UR3lst = [I3,Q3,U3,V3]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,URlst[i],label='Exercise 2 '+Stokeslabel[i])
    plt.plot(x,UR3lst[i],label='Exercise 3 '+Stokeslabel[i])
    plt.legend()
plt.title('Exercise 3')
plt.savefig('ExerciseW3_3.pdf')
plt.show()