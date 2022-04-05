import numpy as np
import matplotlib.pyplot as plt

def planck(nu,T):
    """
    nu: frequency
    T: temperature
    Calculates the value of the planck function for different frequencies and temperatures.
    """
    h=6.626e-34
    e=1.602e-19
    k=1.381e-23
    c=3e8
    B=np.divide(((2*h*nu**3)/(c**2)),(np.exp((h*nu)/(k*T))-1))
    return B

def plancktest(nu_start,nu_stop,T):
    """
    nu_start: starting frequency
    nu_stop: stop frequency
    T: temperature
    Plots the planck function for a temperature T between nu_start and nu_stop. Needs the function planck(nu,T)
    """
    plt.plot(np.arange(nu_start,nu_stop,1e10),planck(np.arange(nu_start,nu_stop,1e10),T))
    plt.show()

def get_ab(T1,tau1,T2,tau2,nu_ref):
    """
    T1: Temperature 1
    T2: Temperature 2
    tau1: Optical depth 1
    tau2: Optical depth 2
    nu_ref: reference frequency
    Approximate the planck function between two different temperatures at a reference frequency as a linear function a*tau +b depending on optical depth. Needs the function planck(nu,T)
    """
    a = (planck(nu_ref,T2)-planck(nu_ref,T1))/(tau2-tau1)
    b = planck(nu_ref,T1)-a*tau1
    return a,b

def intensity(tau,tau_0,a,b,mu):
    """
    tau: optical depth
    tau_0: The optical depth at the orginal point
    a: linear function parameter
    b: linear function parameter
    mu: cos(theta) where theta is the inclanation angle
    Calculates the intesity from the radiative transfer equation assuming the source function as a linear function ax+b
    """
    I_in = 1
    c = mu**-1
    I = np.exp(c*(tau-tau_0))*(I_in-a*c*(tau_0+mu)-b)+a*c*(tau+mu)+b
    return I

if __name__ == '__main__':

    nu_ref = 3e8*(1/(500e-9))

    nu_start = 200e12
    nu_stop = 600e12

    tau0 = 0
    tau1=1
    tau2=2

    T1 = 5700
    T2 = 6000

    a,b=get_ab(T1,tau1,T2,tau2,nu_ref)
    intensity1 = intensity(tau1,tau0,a,b,0.5)
    intensity2 = intensity(tau2,tau0,a,b,0.5)

    print(' ')
    print(intensity1)
    print(intensity2)

    plancktest(nu_start,nu_stop,T1)

