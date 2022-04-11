from Tranpol import Zeemansplittest, UR
import numpy as np
import matplotlib.pyplot as plt
import math as m

space = 0.3

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
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.legend(prop = {'size':4})
plt.suptitle('Exercise 2')
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.savefig('ExerciseW3_2.pdf')
plt.show()

#Exercise 3

I3,Q3,U3,V3 = UR(a,b,10,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)
I3000,Q3000,U3000,V3000 = UR(a,b,10000,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)
UR10000lst = UR(a,b,10000,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,2,3,2,2,2,3)

UR3lst = [I3,Q3,U3,V3]
UR3000lst = [I3000,Q3000,U3000,V3000]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,URlst[i],label='Exercise 2 '+Stokeslabel[i])
    plt.plot(x,UR3lst[i],label='Exercise 3 '+Stokeslabel[i]+', B=10G')
    plt.plot(x,UR3000lst[i],label='Exercise 3 '+Stokeslabel[i]+', B=10000G')
    plt.plot(x,UR10000lst[i],label='Exercise 3 '+Stokeslabel[i]+', B=10000G but 5D2-7D3')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.legend(prop = {'size':4})
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.suptitle('Exercise 3')
plt.savefig('ExerciseW3_3.pdf')
plt.show()

#I has its minima at another location and Q,U,V is 0

#Exercise 4

I40,Q40,U40,V40 = UR(a,b,B,x,l_0,ddopller,0,0.05,m.radians(45),m.radians(0),10,2,2,1,2,2,2)
I445,Q445,U445,V445 = UR(a,b,B,x,l_0,ddopller,0,0.05,m.radians(45),m.radians(45),10,2,2,1,2,2,2)
I490,Q490,U490,V490 = UR(a,b,B,x,l_0,ddopller,0,0.05,m.radians(45),m.radians(90),10,2,2,1,2,2,2)

UR40lst = [I40,Q40,U40,V40]
UR445lst = [I445,Q445,U445,V445]
UR490lst = [I490,Q490,U490,V490]

for i in range(4):
    plt.subplot(2,2,i+1)
    #plt.plot(x,URlst[i],label='Exercise 2 '+Stokeslabel[i])
    #plt.plot(x,UR3lst[i],label='Exercise 3 '+Stokeslabel[i])
    plt.plot(x,UR40lst[i],label='Exercise 4, Xi=0 '+Stokeslabel[i])
    plt.plot(x,UR445lst[i],label='Exercise 4, Xi=45 '+Stokeslabel[i])
    plt.plot(x,UR490lst[i],label='Exercise 4, Xi=90 '+Stokeslabel[i])
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.legend(prop = {'size':4})
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.suptitle('Exercise 4')
plt.savefig('ExerciseW3_4.pdf')
plt.show()

#In I and V they don't depend on Xi but in Q and U they do

#Exercise 5

I50,Q50,U50,V50 = UR(a,b,B,x,l_0,ddopller,0,0.05,m.radians(0),m.radians(30),10,2,2,1,2,2,2)
I545,Q545,U545,V545 = UR(a,b,B,x,l_0,ddopller,0,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)
I590,Q590,U590,V590 = UR(a,b,B,x,l_0,ddopller,0,0.05,m.radians(90),m.radians(30),10,2,2,1,2,2,2)
UR5180lst = UR(a,b,B,x,l_0,ddopller,0,0.05,m.radians(180),m.radians(30),10,2,2,1,2,2,2)

UR50lst = [I50,Q50,U50,V50]
UR545lst = [I545,Q545,U545,V545]
UR590lst = [I590,Q590,U590,V590]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,UR50lst[i],label='Exercise 5, Theta=0 '+Stokeslabel[i])
    plt.plot(x,UR545lst[i],label='Exercise 5, Theta=45 '+Stokeslabel[i])
    plt.plot(x,UR590lst[i],label='Exercise 5, Theta=90 '+Stokeslabel[i])
    plt.plot(x,UR5180lst[i],label='Exercise 5, Theta=180 '+Stokeslabel[i])
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.legend(prop = {'size':4})
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.suptitle('Exercise 4')
plt.savefig('ExerciseW3_5.pdf')
plt.show()

#Same as in exercise 4

#Exercise 7

#Changing v_LOS
Iv1,Qv1,Uv1,Vv1 = UR(a,b,B,x,l_0,ddopller,1,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)
Iv20,Qv20,Uv20,Vv20 = UR(a,b,B,x,l_0,ddopller,20,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)

URv1lst = [Iv1,Qv1,Uv1,Vv1]
URv20lst = [Iv20,Qv20,Uv20,Vv20]

#Changing ddopller
Idopller01,Qdopller01,Udopller01,Vdopller01 = UR(a,b,B,x,l_0,0.1,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)
Idopller05,Qdopller05,Udopller05,Vdopller05 = UR(a,b,B,x,l_0,0.5,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)
Idopller1,Qdopller1,Udopller1,Vdopller1 = UR(a,b,B,x,l_0,1,10,0.05,m.radians(45),m.radians(30),10,2,2,1,2,2,2)

URdopller01lst = [Idopller01,Qdopller01,Udopller01,Vdopller01]
URdopller05lst = [Idopller05,Qdopller05,Udopller05,Vdopller05]
URdopller1lst = [Idopller1,Qdopller1,Udopller1,Vdopller1]

#Changing damping constant
Ia02,Qa02,Ua02,Va02 = UR(a,b,B,x,l_0,ddopller,10,0.2,m.radians(45),m.radians(30),10,2,2,1,2,2,2)
Ia05,Qa05,Ua05,Va05 = UR(a,b,B,x,l_0,ddopller,10,0.5,m.radians(45),m.radians(30),10,2,2,1,2,2,2)
Ia1,Qa1,Ua1,Va1 = UR(a,b,B,x,l_0,ddopller,10,1,m.radians(45),m.radians(30),10,2,2,1,2,2,2)

URa02lst = [Ia02,Qa02,Ua02,Va02]
URa05lst = [Ia05,Qa05,Ua05,Va05]
URa1lst = [Ia1,Qa1,Ua1,Va1]

#Plotting
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,URlst[i],label=Stokeslabel[i])
    plt.plot(x,URv1lst[i],label = 'v_LOS=0.1, '+Stokeslabel[i])
    plt.plot(x,URv20lst[i],label = 'v_LOS=20, '+Stokeslabel[i])
    plt.legend(prop = {'size':4})
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.suptitle('Changing v_LOS')
plt.savefig('v_LOS.pdf')
plt.show()


for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,URlst[i],label=Stokeslabel[i])
    plt.plot(x,URdopller01lst[i],label = 'ddopller=0.1, '+Stokeslabel[i])
    plt.plot(x,URdopller05lst[i],label = 'ddopller=0.5, '+Stokeslabel[i])
    plt.plot(x,URdopller1lst[i],label = 'ddopller=1, '+Stokeslabel[i])
    plt.legend(prop = {'size':4})
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.suptitle('Changing ddopller')
plt.savefig('ddopller.pdf')
plt.show()

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,URlst[i],label=Stokeslabel[i])
    plt.plot(x,URa02lst[i],label = 'damping = 0.2, '+Stokeslabel[i])
    plt.plot(x,URa05lst[i],label = 'damping = 0.5, '+Stokeslabel[i])
    plt.plot(x,URa1lst[i],label = 'damping = 1, '+Stokeslabel[i])
    plt.legend(prop = {'size':4})
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.suptitle('Changing damping')
plt.savefig('damping.pdf')
plt.show()

#Changing Zeeman components

#5D2-7D3

#Plotting Zeeman components

Zeemansplittest(2,3,2,2,2,3)

#Changing v_LOS
Iv1,Qv1,Uv1,Vv1 = UR(a,b,B,x,l_0,ddopller,1,0.05,m.radians(45),m.radians(30),10,2,3,2,2,2,3)
Iv20,Qv20,Uv20,Vv20 = UR(a,b,B,x,l_0,ddopller,20,0.05,m.radians(45),m.radians(30),10,2,3,2,2,2,3)

URv1lst = [Iv1,Qv1,Uv1,Vv1]
URv20lst = [Iv20,Qv20,Uv20,Vv20]

#Changing ddopller
Idopller01,Qdopller01,Udopller01,Vdopller01 = UR(a,b,B,x,l_0,0.1,10,0.05,m.radians(45),m.radians(30),10,2,3,2,2,2,3)
Idopller05,Qdopller05,Udopller05,Vdopller05 = UR(a,b,B,x,l_0,0.5,10,0.05,m.radians(45),m.radians(30),10,2,3,2,2,2,3)
Idopller1,Qdopller1,Udopller1,Vdopller1 = UR(a,b,B,x,l_0,1,10,0.05,m.radians(45),m.radians(30),10,2,3,2,2,2,3)

URdopller01lst = [Idopller01,Qdopller01,Udopller01,Vdopller01]
URdopller05lst = [Idopller05,Qdopller05,Udopller05,Vdopller05]
URdopller1lst = [Idopller1,Qdopller1,Udopller1,Vdopller1]

#Changing damping constant
Ia02,Qa02,Ua02,Va02 = UR(a,b,B,x,l_0,ddopller,10,0.2,m.radians(45),m.radians(30),10,2,3,2,2,2,3)
Ia05,Qa05,Ua05,Va05 = UR(a,b,B,x,l_0,ddopller,10,0.5,m.radians(45),m.radians(30),10,2,3,2,2,2,3)
Ia1,Qa1,Ua1,Va1 = UR(a,b,B,x,l_0,ddopller,10,1,m.radians(45),m.radians(30),10,2,3,2,2,2,3)

URa02lst = [Ia02,Qa02,Ua02,Va02]
URa05lst = [Ia05,Qa05,Ua05,Va05]
URa1lst = [Ia1,Qa1,Ua1,Va1]

#Plotting
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,URlst[i],label=Stokeslabel[i])
    plt.plot(x,URv1lst[i],label = 'v_LOS=0.1, '+Stokeslabel[i])
    plt.plot(x,URv20lst[i],label = 'v_LOS=20, '+Stokeslabel[i])
    plt.legend(prop = {'size':4})
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.suptitle('Changing v_LOS')
plt.savefig('v_LOS2.pdf')
plt.show()


for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,URlst[i],label=Stokeslabel[i])
    plt.plot(x,URdopller01lst[i],label = 'ddopller=0.1, '+Stokeslabel[i])
    plt.plot(x,URdopller05lst[i],label = 'ddopller=0.5, '+Stokeslabel[i])
    plt.plot(x,URdopller1lst[i],label = 'ddopller=1, '+Stokeslabel[i])
    plt.legend(prop = {'size':4})
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.suptitle('Changing ddopller')
plt.savefig('ddopller2.pdf')
plt.show()

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,URlst[i],label=Stokeslabel[i])
    plt.plot(x,URa02lst[i],label = 'damping = 0.2, '+Stokeslabel[i])
    plt.plot(x,URa05lst[i],label = 'damping = 0.5, '+Stokeslabel[i])
    plt.plot(x,URa1lst[i],label = 'damping = 1, '+Stokeslabel[i])
    plt.legend(prop = {'size':4})
plt.subplots_adjust(wspace = space + 0.1, hspace = space)
plt.suptitle('Changing damping')
plt.savefig('damping2.pdf')
plt.show()

I,Q,U,V = UR(a,b,B,x,l_0,ddopller,10,0.05,m.radians(45),m.radians(30),10,1,0,3,2,2,2)

URlst = [I,Q,U,V]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(x,URlst[i],label=Stokeslabel[i])
    plt.legend(prop = {'size':4})
plt.suptitle('Delta J = 0')
plt.savefig('DeltaJ0.pdf')
plt.show()