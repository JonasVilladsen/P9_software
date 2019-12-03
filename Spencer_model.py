#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:42:08 2019

@author: Jonas
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
pi = np.pi

class SPGP():
    """
    Channel model given the parameters
    The class servers the purpose of simulating a SISO channel given
    the parameters.
    """
    
    def __init__(self, Lambda_cl = 1*1E7, Lambda_ray=1E8,\
                 gamma_cl = 1E-8, gamma_ray = 5*1E-9,Q = 5*1E-8,sig_w = np.sqrt(26), I = 801,B = 4*1E9, mode = 'rad'):
        """
        
        Parameters
        ----------
        
        Lambda_cl : float
            parameter assiciated with the cluster intensity

        Lambda_ray : float
            parameter assiciated with the ray intensity
        
        gamma_cl : float
            decay rate of the clusters
        
        gamma_ray : float
            decay rate of the rays
            
        Q: float
            Average power of the first arrivals
        
        I : integer
            number of points 
        
        B : float
            Bandwidth
        
        """
        self.Lambda_cl = Lambda_cl
        self.Lambda_ray = Lambda_ray
        self.gamma_cl = gamma_cl
        self.gamma_ray = gamma_ray 
        self.B = B
        self.I = I
        self.t_max = (I-1)/B
        self.t_0 = 0
        self.Q = Q
        self.sig_w = sig_w
        self.delta_f = 1/self.t_max
        self.linspace_t = np.linspace(0,self.t_max,I)
        self.linspace_angle = np.linspace(0,2*pi,I)

        self.delta_f_tilde = 1/(2*pi)
            
        
        
    def PPP(self,intensity,start):
        """
        Samples a Poisson point process by sampling the number of points from
        a poisson and uniformly distrubting them
        """
        TDelta=self.t_max-start
        #Simulate Poisson point process
        numbPoints = scipy.stats.poisson(intensity*TDelta).rvs()#Poisson counting process
        PPP = TDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))#Uniformly distribute the number of points
        return np.concatenate(np.vstack([0,PPP]))
    
    
    def beta(self,T,tau):
        """
        Simulates the complex gain
        """
        sig = (self.Q/2*np.exp(-T/self.gamma_cl-tau/self.gamma_ray))
        
        real = np.random.normal(0,np.sqrt(sig))
        im = 1j*np.random.normal(0,np.sqrt(sig))
        return real+im
    

    def Theta(self):
        uniform = np.random.uniform(0,2*pi,size = len(self.T)-1)
        return np.concatenate(np.vstack([np.pi,(uniform.reshape(len(self.T)-1,1))]))
       
    def omega(self, size):
            return np.deg2rad(np.random.laplace(0,self.sig_w,size))
    
    
    def simulate_SP(self):
        H = np.zeros(shape=(self.I,self.I),dtype = np.complex)
        T = self.PPP(self.Lambda_cl,0)
        self.T = T
        self.Angle_cl = self.Theta()
        self.angle_sum = []
        self.delay_sum = []
        arange = np.arange(self.I)
        for l in range(len(T)):
            tau = self.PPP(self.Lambda_ray,T[l])
            angle_ray = self.omega(len(tau))
            self.angle_sum.append(self.Angle_cl[l]+angle_ray)
            self.delay_sum.append(self.T[l]+tau)
            for k in range(len(tau)):
                beta_l = self.beta(T[l],tau[k])

                delay = (self.delta_f*arange*(T[l]+tau[k])).reshape(self.I,1)
                angle = (self.delta_f_tilde*arange*(self.Angle_cl[l]+angle_ray[k])).reshape(1,self.I)
                
                H += beta_l*np.exp(-1j*2*pi*(delay+angle))
        self.H = H
        return H,self.linspace_t,self.linspace_angle
    
    
    def plots(self,save = False):
        H,t,a = self.simulate_SP()
        t_tilde = []
        for j in range(len(t)):
            t_tilde.append(np.format_float_scientific(np.float32(t[j]), exp_digits=2))
        a = np.around(a,2)
        h = np.fft.ifft2(H)
        color_list = ['b','g','r','c','m','y']
        plots,plots1 = [],[]
        plt.figure()
        for j in range(len(self.T)):
            plots.append(plt.plot(self.T[j],self.Angle_cl[j],color_list[j]+'o', markersize=12,label = r'$(T_l, \Theta_l)$',alpha = 0.25))
 
        for j in range(len(self.T)):
            plots1.append(plt.plot(self.delay_sum[j],self.angle_sum[j],color_list[j] + 'x', label = r'$(T_l + \tau_{l,k},\Theta_l + \omega_{l,k})$'))
        plt.xlabel('Delay [s]')
        plt.ylabel('Angle [Degrees]')
        plt.legend([plots,plots1],['1','2'])
        plt.ylim([0,360])
        plt.tight_layout()

        if save == True:
            plt.savefig('Pictures/spencer_marks.png',dpi = 500)
            
        
        plt.figure(figsize = (8.25, 5))
        step = 400 #int((self.I-1)/len(self.x_ticks[0]))
        plt.imshow(20*np.log10(abs(h).T),origin = 'lower', aspect='auto')
        plt.colorbar()
        plt.xlabel('Delay [s]')
        plt.ylabel('Angle [Degrees]')
        plt.xticks(np.arange(self.I)[::step],t_tilde[::step])
        plt.yticks([0,step,2*step],[0,180,360])
        plt.tight_layout()
        if save == True:
            plt.savefig('Pictures/spencer_model_dB.png',dpi = 500)
        
        
        




