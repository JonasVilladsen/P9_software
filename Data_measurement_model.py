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

class SPGP_meas():
    """
    Channel model given the parameters
    The class servers the purpose of simulating a SISO channel given
    the parameters.
    """
    
    def __init__(self, Lambda_cl = 1E7, Lambda_ray=1E9,\
                 gamma_cl = 1E-8, gamma_ray = 5*1E-9,Q = 5*1E-8,\
                 sig_w = np.sqrt(26),I = 801,B = 4*1E9,n = 100,d = 0.1):
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
        self.linspace = np.linspace(0,self.t_max,I)
        self.wave_len = 3E5/60E9 # c/f_C
        self.n = n
        self.d = self.wave_len/2 # lambda/2
        self.r = self.construct_r()
        
        
    def PPP(self,intensity,start):
        """
        Samples a Poisson point process by sampling the number of points from
        a poisson and uniformly distrubting them
        """
        TDelta=self.t_max-start
        #Simulate Poisson point process
        numbPoints = scipy.stats.poisson(intensity*TDelta).rvs()#Poisson counting process
        PPP = TDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))#Uniformly distribute the number of points
        return (np.concatenate(np.vstack([0,PPP])))
    
    
    def beta(self,T,tau):
        """
        Simulates the complex gain
        """
        sig = (self.Q/2*np.exp(-T/self.gamma_cl-tau/self.gamma_ray))
        
        real = np.random.normal(0,np.sqrt(sig))
        im = 1j*np.random.normal(0,np.sqrt(sig))
        return real+im
    
    def G(self,angle):
        return 1
    
    def construct_r(self):
        """
        Constructs r_n for ULA
        """
        r = []
        for j in range(self.n):
            r.append(np.array([j*self.d,0]))
        return r
    
    def omega(self, size):
        self.sig_w = self.sig_w
        return np.deg2rad(np.random.laplace(0,self.sig_w,size))
    
    def Theta(self):
        uniform = np.random.uniform(0,pi,size = len(self.T)-1)
        theta0 = np.pi/2
        return np.concatenate(np.vstack([theta0,(uniform.reshape(len(self.T)-1,1))]))
   
    def angle_func(self,summed_ang):
        return np.array([np.cos(summed_ang),np.sin(summed_ang)])
    
    
    def simulate_meas(self,Q = None,sig = None):
        """
        Simulation of equation  (3.19)
        """
        H_list = []
        if sig != None:
            self.sig_w = sig
        if Q != None:
            self.Q = Q
        T = self.PPP(self.Lambda_cl,0)
        self.T = T
        self.tau_list =[]
        self.angle_list = []
        self.beta_list = [[] for j in range(len(T))]
        r = self.construct_r()
        self.Angle_cl = self.Theta()
        H = np.zeros(shape=(self.I),dtype = np.complex)
        for l in range(len(T)):
            tau = self.PPP(self.Lambda_ray,T[l])
            self.tau_list.append(tau)
            angle_ray = self.omega(len(tau))
            self.angle_list.append(angle_ray)
            for k in range(len(tau)):
                beta_l = self.beta(T[l],tau[k])
                self.beta_list[l].append(beta_l)
                angle = np.exp(-1j*2*(pi/self.wave_len)\
                        *np.dot(self.angle_func(self.Angle_cl[l]+angle_ray[k]),r[0]))
                H += beta_l*np.exp(-1j*2*pi*self.delta_f*np.arange(self.I)*(T[l]+tau[k]))*angle*self.G(self.Angle_cl[l]+angle_ray[k])
        H_list.append(H)
        
        self.test = []
        for n in range(1,self.n):
            H = np.zeros(shape=(self.I),dtype = np.complex)
            for l in range(len(T)):
                for k in range(len(self.tau_list[l])):
                    angle_exp = np.exp(-1j*2*(pi/self.wave_len)\
                        *np.dot(self.angle_func(self.Angle_cl[l]+self.angle_list[l][k]),r[n]))
                    H += self.beta_list[l][k]*np.exp(-1j*2*pi*self.delta_f*np.arange(self.I)*\
                     (T[l]+self.tau_list[l][k]))*angle_exp*self.G(self.Angle_cl[l]+self.angle_list[l][k])
            H_list.append(H)
        return np.asarray(H_list)
    
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def plots(self,save = False):
        H = self.simulate_meas()
        
        for j in range(self.n):
            plt.plot(self.linspace,20*np.log10(abs(np.fft.ifft(H[j]))),label = r'$20\log_{10}\vert y^{(%d)}(t)\vert$'%j,alpha = 0.5)
        plt.title('Spencer measurement model for %d antennas' %self.n)
        plt.ylabel('Power [dB]')
        plt.legend()
        plt.tight_layout()
        if save == True:
            plt.savefig('Pictures/Data_meas_model_d{}.png'.format(self.d),dpi = 500)
