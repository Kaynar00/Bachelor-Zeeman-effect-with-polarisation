from re import A
import Transfer as tr
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import math as m

def lv_convert(l):
    '''
    l: wavelength
    takes a wavelength of ligth l and converts it to frequency
    '''
    c = 3e8
    v = c/l
    return v

def g(S,L,J):
    '''
    S: Spin quantum number
    L: Angular momentum quantum number
    J: Total angular momentum quantum number
    Calculates the Landé factor for different states
    '''
    if J == 0:
        g=0
        return g
    if J != 0:
        g = 3/2 + ((S*(S+1)-(L+1)*L)/(2*J*(J+1)))
        return g

def g_eff(S_l,L_l,J_l,S_u,L_u,J_u):
    '''
    S_l: Lower state spin quantum number
    L_l: Lower state angular momentum quantum number
    J_l: Lower state total angular momentum quantum number
    S_u: Upper state spin quantum number
    L_l: Upper state angular momentum quantum number
    J_l: Upper state total angular momentum quantum number
    Calculates the effective Landé factor between different states
    '''
    return 0.5*(g(S_l,L_l,J_l)+g(S_u,L_u,J_u))+0.25*(g(S_l,L_l,J_l)-g(S_u,L_u,J_u))*((J_l*(J_l+1))-(J_u*(J_u+1)))

def v(l,l_0,delta_l):
    '''
    l: wavelenght
    l_0: wavelength for the line
    delta_l
    '''
    return (l-l_0)/(delta_l)

def v_A(l_0,v_LOS,delta_l):
    c=3e8 #m/s
    return (l_0*v_LOS)/(c*delta_l)

def v_B(l_0,delta_l,B,S_i,L_i,J_i,S_f,L_f,J_f):
    g_i = g(S_i,L_i,J_i)
    g_f = g(S_f,L_f,J_f)
    return (4.67e-13)*((l_0**2)/(delta_l))*

def zcomp(M,Mp,J,Jp):
    '''
    M: Initial magnetic quantum number
    Mp: Final magnetic quantum number
    J: Initial rotational quantum number
    Jp: Final rotational quantum nuber
    Calculates the Zeeman components for different M, Mp ,J and Jp.
    '''
    sigma_b = False
    pi = False
    sigma_r = False
    if (Mp-M) == 1 and (Jp-J) == 1:
        z = (3*(J+M+1)*(J+M+2))/(2*(J+1)*(2*J+1)*(2*J+3))
        print('sigma_b state')
        sigma_b = True
        return z, sigma_b, pi, sigma_r
    elif (Mp-M) == 1 and (Jp-J) == 0:
        z = (3*(J-M)*(J+M+1))/(2*J*(J+1)*(2*J+1))
        print('sigma_b state')
        sigma_b = True
        return z, sigma_b, pi, sigma_r
    elif (Mp-M) == 1 and (Jp-J) == -1:
        z = (3*(J-M)*(J-M-1))/(2*J*(2*J-1)*(2*J+1))
        print('sigma_b state')
        sigma_b = True
        return z, sigma_b, pi, sigma_r
    elif (Mp-M) == 0 and (Jp-J) == 1:
        z = (3*(J-M+1)*(J+M+1))/((J+1)*(2*J+1)*(2*J+3))
        print('pi state')
        pi = True
        return z, sigma_b, pi, sigma_r
    elif (Mp-M) == 0 and (Jp-J) == 0:
        z = (3*M**2)/(J*(J+1)*(2*J+1))
        print('pi state')
        pi = True
        return z, sigma_b, pi, sigma_r
    elif (Mp-M) == 0 and (Jp-J) == -1:
        z = (3*(J-M)*(J+M))/(J*(2*J-1)*(2*J+1))
        print('pi state')
        pi = True
        return z, sigma_b, pi, sigma_r
    elif (Mp-M) == -1 and (Jp-J) == 1:
        z = (3*(J-M+1)*(J-M+2))/(2*(J+1)*(2*J+1)*(2*J+3))
        print('sigma_r state')
        sigma_r = True
        return z, sigma_b, pi, sigma_r
    elif (Mp-M) == -1 and (Jp-J) == 0:
        z = (3*(J+M)*(J-M+1))/(2*J*(J+1)*(2*J+1))
        print('sigma_r state')
        sigma_r = True
        return z, sigma_b, pi, sigma_r
    elif (Mp-M) == -1 and (Jp-J) == -1:
        z = (3*(J+M)*(J+M-1))/(2*J*(2*J-1)*(2*J+1))
        print('sigma_r state')
        sigma_r = True
        return z, sigma_b, pi, sigma_r
    else:
        print('That is a forbidden transition')

def Zeemansplit(J_l,J_u,L_l,L_u,S_l,S_u):
    g_l = g(S_l,L_l,J_l)
    g_u = g(S_u,L_u,J_u)

    M_l = list(range(-J_l,J_l+1))
    M_u = list(range(-J_u,J_u+1))

    split = []
    comp = []
    sig_b_lst = []
    pi_lst = []
    sig_r_lst = []

    for l in M_l:
        for u in M_u:
            if l == u + 1 or l == u or l == u-1:
                s = g_l*l-g_u*u
                c, sigma_b, pi, sigma_r = zcomp(l,u,J_l,J_u)
                split.append(s)
                comp.append(c)
                sig_b_lst.append(sigma_b)
                pi_lst.append(pi)
                sig_r_lst.append(sigma_r)
    return split, comp,sig_b_lst, pi_lst, sig_r_lst

def Zeemansplittest(J_l,J_u,L_l,L_u,S_l,S_u):
    split, comp, sigma_b, pi, sigma_r = Zeemansplit(J_l,J_u,L_l,L_u,S_l,S_u)
    newfig = plt.figure()
    for i in range(len(split)):
        if sigma_b[i] == True:
            plt.bar(split[i],comp[i],width = 0.1,color = 'b')
        elif pi[i] == True:
            plt.bar(split[i],comp[i],width = 0.1,color = 'g')
        elif sigma_r[i] == True:
            plt.bar(split[i],comp[i],width = 0.1,color = 'r')
    plt.savefig('Zeemansplit.png')
    plt.show()

def profiles(v,v_A,v_B,g_i,g_f,M_i,M_f,J_i,J_f,aimag):
    '''
    v: reduced frequency
    v_A: the normalized shift due to bulking
    v_B: the normalized zeeman splitting
    J_i: the rotational quantum nuber for the initial state
    J_f: the rotational quantum number for the final state
    aimag: the damping constant times the imaginary number "j" 
    calculates the profile value at different frequencies
    '''
    z = special.wofz(v-v_A+v_B*(g_f*M_f-g_i*M_i)+1j*aimag)
    f, sigma_b, pi, sigma_r = zcomp(M_i,M_f,J_i,J_f)
    eta = f*z.real
    rho = f*z.imag
    return eta, rho, sigma_b, pi, sigma_r

def profilestest(v,v_A,v_B,g_i,g_f,M_i,M_f,J_i,J_f,aimag):
    '''
    v: reduced frequency
    v_A: the normalized shift due to bulking
    v_B: the normalized Zeeman splitting
    J_i: the rotational quantum nuber for the initial state
    J_f: the rotational quantum number for the final state
    aimag: the damping constant times the imaginary number "j"
    plots the function profiles for different frequencies
    '''
    eta, rho, sigma_b, pi, sigma_r = profiles(v,v_A,v_B,g_i,g_f,M_i,M_f,J_i,J_f,aimag)
    newfig = plt.figure()
    plt.plot(v,eta,label='eta')
    plt.plot(v,rho,label='rho')
    plt.xlabel('v')
    plt.legend()
    plt.savefig('profiles.png')
    plt.show()

def profilessum(v,v_A,v_B,J_i,J_f,L_i,L_f,S_i,S_f,aimag):
    '''
    v: reduced frequency
    v_A: the normalized shift due to bulking
    v_B: the normalized Zeeman shifting
    J_i: the rotational quantum nuber for initial state
    J_f: the rotational quantumb number for final state
    L_i: the angular momentum quantum number for initial state
    L_f: the angular momentum quantum number for final state
    S_i: the spin quantum number for initial state
    S_f: the spin quantum number for final state
    aimag: the damping constatn times the imaginary number "j"
    Calculates the profiles eta_b, eta_p, eta_r, rho_b, rho_p and rho_r
    '''
    print(J_i)
    M_i = list(range(-J_i,J_i+1))
    M_f = list(range(-J_f,J_f+1))
    eta_b = 0
    eta_p = 0
    eta_r = 0
    rho_b = 0
    rho_p = 0
    rho_r = 0

    for i in M_i:
        for f in M_f:
            if i == f + 1 or i == f or i == f-1:
                g_i = g(S_i,L_i,J_i)
                g_f = g(S_f,L_f,J_f)
                eta, rho, sigma_b, pi, sigma_r = profiles(v,v_A,v_B,g_i,g_f,i,f,J_i,J_f,aimag)
                if sigma_b == True:
                    eta_b += eta
                    rho_b += rho
                elif pi == True:
                    eta_p += eta
                    rho_p += rho
                elif sigma_r == True:
                    eta_r += eta
                    rho_r += rho
    return eta_b, eta_p, eta_r, rho_b, rho_p, rho_r

def profilessumtest(v,v_A,v_B,J_i,J_f,L_i,L_f,S_i,S_f,aimag):
    '''
    v: reduced frequency
    v_A: the normalized shift due to bulking
    v_B: the normalized Zeeman shifting
    J_i: the rotational quantum nuber for initial state
    J_f: the rotational quantumb number for final state
    L_i: the angular momentum quantum number for initial state
    L_f: the angular momentum quantum number for final state
    S_i: the spin quantum number for initial state
    S_f: the spin quantum number for final state
    aimag: the damping constatn times the imaginary number "j"
    Plots the profiles eta_b, eta_p, eta_r, rho_b, rho_p and rho_r depending on different relative frequencies
    '''
    eta_b, eta_p, eta_r, rho_b, rho_p, rho_r = profilessum(v,v_A,v_B,J_i,J_f,L_i,L_f,S_i,S_f,aimag)
    newfig = plt.figure()
    plt.plot(v,eta_b,label='eta_b')
    plt.plot(v,eta_p,label='eta_p')
    plt.plot(v,eta_r,label='eta_r')
    plt.plot(v,rho_b,label='rho_b')
    plt.plot(v,rho_p,label='rho_p')
    plt.plot(v,rho_r,label='rho_r')
    plt.xlabel('v')
    plt.legend()
    plt.savefig('profilessum.png')
    plt.show()

def trcoefs(v,v_A,v_B,J_i,J_f,L_i,L_f,S_i,S_f,aimag,theta,X,eta_0):
    '''
    v: reduced frequency
    v_A: the normalized shift due to bulking
    v_B: the normalized Zeeman shifting
    J_i: the rotational quantum nuber for initial state
    J_f: the rotational quantumb number for final state
    L_i: the angular momentum quantum number for initial state
    L_f: the angular momentum quantum number for final state
    S_i: the spin quantum number for initial state
    S_f: the spin quantum number for final state
    aimag: the damping constatn times the imaginary number "j"
    theta: an angle for polarisation
    X: an angle for polarisation
    Caclulates the transition coefficients
    '''
    eta_b, eta_p, eta_r, rho_b, rho_p, rho_r = profilessum(v,v_A,v_B,J_i,J_f,L_i,L_f,S_i,S_f,aimag)
    hI = 1+eta_0*0.5*(eta_p*m.sin(theta)**2+((eta_b+eta_r)/2)*(1+m.cos(theta)**2))
    hQ = eta_0**0.5*(eta_p-((eta_b+eta_r)/2))*m.sin(theta)**2*m.cos(2*X)
    hU = eta_0*0.5*(eta_p-((eta_b+eta_r)/2))*m.sin(theta)**2*m.sin(2*X)
    hV = eta_0*0.5*(eta_r-eta_b)*m.cos(theta)
    rQ = eta_0*0.5*(rho_p-((rho_b+rho_r)/2))*m.sin(theta)**2*m.cos(2*X)
    rU = eta_0*0.5*(rho_p-((rho_b+rho_r)/2))*m.sin(theta)**2*m.sin(2*X)
    rV = eta_0*0.5*(rho_r-rho_b)*m.cos(theta)
    return hI, hQ, hU, hV, rQ, rU, rV

def trcoefstest(v,v_A,v_B,J_i,J_f,L_i,L_f,S_i,S_f,aimag,theta,X,eta_0):
    '''
    v: reduced frequency
    v_A: the normalized shift due to bulking
    v_B: the normalized Zeeman shifting
    J_i: the rotational quantum nuber for initial state
    J_f: the rotational quantumb number for final state
    L_i: the angular momentum quantum number for initial state
    L_f: the angular momentum quantum number for final state
    S_i: the spin quantum number for initial state
    S_f: the spin quantum number for final state
    aimag: the damping constatn times the imaginary number "j"
    theta: an angle for polarisation
    X: an angle for polarisation
    Plots the transition coefficients depending on different reduced frequency
    '''
    hI, hQ, hU, hV, rQ, rU, rV = trcoefs(v,v_A,v_B,J_i,J_f,L_i,L_f,S_i,S_f,aimag,theta,X,eta_0)
    newfig = plt.figure
    plt.plot(v,hI,label='hI')
    plt.plot(v,hQ,label='hQ')
    plt.plot(v,hU,label='hU')
    plt.plot(v,hV,label='hV')
    plt.plot(v,rQ,label='rQ')
    plt.plot(v,rU,label='rU')
    plt.plot(v,rV,label='rV')
    plt.xlabel('v')
    plt.legend()
    plt.savefig('transfer_coefficients.png')
    plt.show()

def UR(a,b,v,v_A,v_B,J_i,J_f,L_i,L_f,S_i,S_f,aimag,theta,X,eta_0):
    '''
    T1: Temperature at tau1
    tau1: Optical depth 1
    T2: Temperature at tau2
    tau2: Optical depth 2
    nu_ref: A reference frequency at the continuum
    v: reduced frequency
    v_A: the normalized shift due to bulking
    v_B: the normalized Zeeman shifting
    J_i: the rotational quantum nuber for initial state
    J_f: the rotational quantumb number for final state
    L_i: the angular momentum quantum number for initial state
    L_f: the angular momentum quantum number for final state
    S_i: the spin quantum number for initial state
    S_f: the spin quantum number for final state
    aimag: the damping constatn times the imaginary number "j"
    theta: an angle for polarisation
    X: an angle for polarisation
    Calculates the intensities for the different stokes parameters with the Unno-Rachkovsky solutions.
    '''
    hI, hQ, hU, hV, rQ, rU, rV = trcoefs(v,v_A,v_B,J_i,J_f,L_i,L_f,S_i,S_f,aimag,theta,X,eta_0)
    Pi = hQ*rQ + hU*rU + hV*rV
    delta = (hI**2)*((hI**2)-(hQ**2)-(hV**2)+(rQ**2)+(rU**2)+(rV**2))-Pi**2
    I = b+(delta**-1)*(hI*((hI**2)+(rQ**2)+(rU**2)+(rV**2)))*a
    Q = -(delta**-1)*((hI**2)*hQ+hI*(hV*rU-hU*rV)+rQ*Pi)*a
    U = -(delta**-1)*((hI**2)*hU+hI*(hQ*rV-hV*rQ)+rU*Pi)*a
    V = -(delta**-1)*((hI**2)*hV+hI*(hU*rQ-hQ*rU)+rV*Pi)*a
    return I,Q,U,V

def UR_test(a,b,v,v_A,v_B,J_i,J_f,L_i,L_f,S_i,S_f,aimag,theta,X,eta_0):
    I,Q,U,V = UR(a,b,v,v_A,v_B,J_i,J_f,L_i,L_f,S_i,S_f,aimag,theta,X,eta_0)
    newfig = plt.figure()
    plt.plot(v,I,label='I')
    plt.plot(v,Q,label='Q')
    plt.plot(v,U,label='U')
    plt.plot(v,V,label='V')
    plt.legend()
    plt.savefig('UR_solutions.png')
    plt.show()

if __name__ == '__main__' :  
    nu_start = 0
    nu_stop = lv_convert(6000e-10)
    nu_step = 1e10 

    tau1=1
    tau2=2

    T1 = 5700
    T2 = 6000

    a = 0.8
    b = 0.2
    
    eta, rho, sigma_b, pi, sigma_r = profiles(500,550,450,1,1,1,2,1,2,30)
    print(eta)
    print(rho)

    profilestest(np.arange(100,900),550,450,1,1,1,2,1,2,30)

    profilessumtest(np.arange(nu_start,nu_stop,nu_step),lv_convert(300e-9),lv_convert(5247.1e-10),2,3,2,2,2,3,30j)

    trcoefstest(np.arange(nu_start,nu_stop,nu_step),lv_convert(300e-9),lv_convert(5247.1e-10),2,3,2,2,2,3,30j,m.pi/4,m.pi/4,1)

    UR_test(a,b,np.arange(nu_start,nu_stop,nu_step),lv_convert(300e-9),lv_convert(5247.1e-10),2,3,2,2,2,3,30j,m.pi/4,m.pi/4,1)

    #z = zcomp(0,1,2,1)
    #print(z),30