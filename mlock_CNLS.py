"""
Created on Sat Apr 29 12:33:32 2017

@author: Thomas
"""

# numerical packages
import numpy as np
from numpy.testing import assert_equal
from scipy.integrate import complex_ode
#from scipy.stats import kurtosis
from scipy.stats import moment

# time measurement
import time

"""Definition of the model parameter"""

def laser_simulation(uvt, alpha1, alpha2, alpha3, alphap,K):
    # parameters of the Maxwell equation
    """
    D = 1
    K = 0.1
    E0 = 1
    tau = 0.1
    g0 = 0.3
    Gamma = 0.2
    """
    D = -0.4
    E0 = 4.23
    tau = 0.1
    g0 = 1.73
    Gamma = 0.1
    
    Z = 1.5         # cavity length
    T = 60
    n = 256       # t slices
    Rnd = 500     # round trips
    t2 = np.linspace(-T/2,T/2,n+1)
    t_dis = t2[0:n].reshape([1,n])   # time discretization
    new = np.concatenate((np.linspace(0,n//2-1,n//2),
                          np.linspace(-n//2,-1,n//2)),0)
    k = (2*np.pi/T)*new
    ts=[]
    ys=[]
    t0=0.0
    tend=1
    
    # waveplates & polarizer
    W4 = np.array([[np.exp(-1j*np.pi/4), 0],[0, np.exp(1j*np.pi/4)]]); # quarter waveplate
    W2 = np.array([[-1j, 0],[0, 1j]]);  # half waveplate
    WP = np.array([[1, 0], [0, 0]]);  # polarizer
    
    # waveplate settings
    R1 = np.array([[np.cos(alpha1), -np.sin(alpha1)], 
                   [np.sin(alpha1), np.cos(alpha1)]])
    R2 = np.array([[np.cos(alpha2), -np.sin(alpha2)], 
                   [np.sin(alpha2), np.cos(alpha2)]])
    R3 = np.array([[np.cos(alpha3), -np.sin(alpha3)], 
                   [np.sin(alpha3), np.cos(alpha3)]])
    RP = np.array([[np.cos(alphap), -np.sin(alphap)], 
                   [np.sin(alphap), np.cos(alphap)]])
    J1 = np.matmul(np.matmul(R1,W4),np.transpose(R1))
    J2 = np.matmul(np.matmul(R2,W4),np.transpose(R2))
    J3 = np.matmul(np.matmul(R3,W2),np.transpose(R3))
    JP = np.matmul(np.matmul(RP,WP),np.transpose(RP))
    
    # transfer function
    Transf = np.matmul(np.matmul(np.matmul(J1,JP),J2),J3)
    
    urnd=np.zeros([Rnd, n], dtype=complex)
    vrnd=np.zeros([Rnd, n], dtype=complex)
    t_dis=t_dis.reshape(n,)
    energy=np.zeros([1,Rnd])
    
    # definition of the rhs of the ode
    def mlock_CNLS_rhs(ts, uvt):
        [ut_rhs,vt_rhs] = np.split(uvt,2)
        u = np.fft.ifft(ut_rhs)
        v = np.fft.ifft(vt_rhs)
        # calculation of the energy function
        E = np.trapz(np.conj(u)*u+np.conj(v)*v,t_dis)
        
        # u of the rhs
        urhs = -1j*0.5*D*(k**2)*ut_rhs - 1j*K*ut_rhs + \
                1j*np.fft.fft((np.conj(u)*u+ (2/3)*np.conj(v)*v)*u + \
                              (1/3)*(v**2)*np.conj(u)) + \
                2*g0/(1+E/E0)*(1-tau*(k**2))*ut_rhs - Gamma*ut_rhs
        
        # v of the rhs
        vrhs = -1j*0.5*D*(k**2)*vt_rhs + 1j*K*vt_rhs + \
                1j*np.fft.fft((np.conj(v)*v+(2/3)*np.conj(u)*u)*v + \
                              (1/3)*(u**2)*np.conj(v) ) + \
                2*g0/(1+E/E0)*(1-tau*(k**2))*vt_rhs - Gamma*vt_rhs
         
        return np.concatenate((urhs, vrhs),axis=0)
    
    # definition of the solution output for the ode integration
    def solout(t,y):
        ts.append(t)
        ys.append(y.copy())
        
    start = time.time()
    
    uv_list = []
    norms = []
    change_norm = 100
    jrnd = 0
    # solving the ode for Rnd rounds
    while(jrnd < Rnd and change_norm > 1e-6):
        ts = []
        ys = []
        
        t0 = Z*jrnd
        tend = Z*(jrnd+1)
        
        uvtsol = complex_ode(mlock_CNLS_rhs)
        uvtsol.set_integrator(method='adams', name='dop853') # alternative 'dopri5'
        uvtsol.set_solout(solout)
        uvtsol.set_initial_value(uvt, t0)
        sol = uvtsol.integrate(tend)
        assert_equal(ts[0], t0)
        assert_equal(ts[-1], tend)
        
        u=np.fft.ifft(sol[0:n])
        v=np.fft.ifft(sol[n:2*n])
        
        urnd[jrnd,:]=u
        vrnd[jrnd,:]=v
        energy[0, jrnd]=np.trapz(np.abs(u)**2+np.abs(v)**2,t_dis)
        
        uvplus=np.matmul(Transf,np.transpose(np.concatenate((u.reshape(n,1),
                                                              v.reshape(n,1)),axis=1)))
        uv_list.append(np.concatenate((uvplus[0,:],
                                       uvplus[1,:]), axis=0))
        
        uvt=np.concatenate((np.fft.fft(uvplus[0,:]),
                                       np.fft.fft(uvplus[1,:])), axis=0)
        
        if jrnd > 0:
            phi=np.sqrt(np.abs(np.vstack(uv_list)[:,:n])**2 + \
                        np.abs(np.vstack(uv_list)[:,n:2*n])**2)
            change_norm=np.linalg.norm((phi[-1,:]-phi[len(phi)-2,:]))/ \
            np.linalg.norm(phi[len(phi)-2,:])
            norms.append(change_norm) 
            
        jrnd += 1
    
    
    kur = np.abs(np.fft.fftshift(np.fft.fft(phi[-1,:])))
    #M4 = kurtosis(kur)
    M4 = moment(kur,4)/np.std(kur)**4
    
    end = time.time()
    print(end-start)
    
    E = np.sqrt(np.trapz(phi[-1,:]**2, t_dis))
    
    states = np.array([E, M4, alpha1, alpha2, alpha3, alphap])
    
    """ surface plot 
    # create meshgrid
    X, Y = np.meshgrid(t_dis,np.arange(0,len(norms)))
    
    # figure urnd
    fig_urand = plt.figure()
    ax = fig_urand.gca(projection='3d')
    
    # plot the surface
    surf = ax.plot_surface(X, Y, np.abs(urnd[:len(norms),:]), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Add a color bar which maps values to colors.
    fig_urand.colorbar(surf, shrink=0.5, aspect=5)
    
    
    # figure vrnd
    fig_vrand = plt.figure()
    ax = fig_vrand.gca(projection='3d')
    
    # plot the surface
    surf = ax.plot_surface(X, Y, np.abs(vrnd[:len(norms),:]), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Add a color bar which maps values to colors.
    fig_vrand.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()
    """
    return (uvt,states)
