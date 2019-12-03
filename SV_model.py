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

class SVGP():
    """
    Channel model given the parameters
    The class servers the purpose of simulating a SISO channel given
    the parameters.
    """
    
    def __init__(self, Lambda_cl = 1*1E7, Lambda_ray=1E9,\
                 gamma_cl = 1E-8, gamma_ray = 5*1E-9,Q = 5*1E-8, I = 801,B = 4*1E9):
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
        self.delta_f = 1/self.t_max
        self.linspace = np.linspace(0,self.t_max,I)
        
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
    

    
    def simulate_SV(self,window = 'rect'):
        
        H = np.zeros(shape=(self.I),dtype = np.complex)
        T = self.PPP(self.Lambda_cl,0)
        self.T = T
        self.tau_list =[]
        if window == 'rect':
            for l in range(len(T)):
                tau = self.PPP(self.Lambda_ray,T[l])
                self.tau_list.append(tau)
                for k in range(len(tau)):
                    beta_l = self.beta(T[l],tau[k])
                    H += beta_l*np.exp(-1j*2*pi*self.delta_f*np.arange(self.I)*(T[l]+tau[k])) #
        if window == 'Hann':
            for l in range(len(T)):
                tau = self.PPP(self.Lambda_ray,T[l])
                self.tau_list.append(tau)
                for k in range(len(tau)):
                    beta_l = self.beta(T[l],tau[k])
                    H += beta_l*np.exp(-1j*2*pi*self.delta_f*np.arange(self.I)*(T[l]+tau[k]))*np.hanning(self.I) #
        self.H = H   
        return H,self.linspace
    
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    

    def plot_SV(self,scale = 'Normal',save = 'False'):
        
        self.idx = []
        for j in range(len(self.T)):
            self.idx.append(self.find_nearest(self.linspace,self.T[j]))
        
        y_abs = np.abs(np.fft.ifft(self.H))
        y_abs_scale = np.abs(np.fft.ifft(self.H))
        y_scale = y_abs_scale
        
        last_idx = 2
        plt.figure()
        plt.title('SV model')
        plt.plot(self.linspace[:-last_idx],y_abs[:-last_idx]**2,label = r'$\vert y(t) \vert^2$')
        plt.plot(self.linspace[self.idx],(y_abs[self.idx]**2),'x',label = r'$T_l$')
        plt.ticklabel_format(axis ='y',scilimits=(0,0))
        plt.xlabel('Delay [s]')
        plt.ylabel(r'Power')
        plt.legend()
        plt.tight_layout()
        if save == 'True':
            plt.savefig('Pictures/model_G_cl{}_G_r{}.png'.format(self.gamma_cl,self.gamma_ray),dpi = 500)

        plt.figure()
        plt.title( 'SV model')
        plt.plot(self.linspace[:-last_idx],20*np.log10(y_scale)[:-last_idx],label = r'$10\log_{10}\vert y(t) \vert^2$')
        plt.plot(self.linspace[self.idx],20*np.log10(y_scale[self.idx]),'x',label = r'$T_l$')
        plt.ylabel(r'$Power [dB]$')
        plt.ticklabel_format(axis ='y',scilimits=(0,0))
        plt.xlabel('Delay [s]')
        plt.legend()
        plt.tight_layout()
        if save == 'True':
            plt.savefig('Pictures/model_Log_G_cl{}_G_r{}.png'.format(self.gamma_cl,self.gamma_ray),dpi = 500)

    def plot_SV_power_intensity(self,N_sim = 1000,save = 'False', window = 'rect'):
        y_abs_scale = []
        
        for n in range(N_sim):
            y_abs_scale.append(np.abs(np.fft.ifft(self.simulate_SV(window=window)[0]))**2)
        
        y_mean = np.mean(np.asarray(y_abs_scale),axis = 0)
        
        self.P_t_dB = 10*np.log10(self.delay_power_intensity()[0])-10*np.log10(self.B) # We have to take the bandwidth into account
        t = self.delay_power_intensity()[1]
        self.y_mean_dB = 10*np.log10(y_mean)

        
        plt.figure()
        plt.title('Estimated delay-power intensity, window : {}'.format(window))
        
        plt.plot(t[:400], self.P_t_dB[:400],label = 'Theoretical APDP [dB]')
        plt.plot(self.linspace[:400],self.y_mean_dB[:400],label = r'Estimated APDP [dB]')
        plt.ylabel(r'$Power [dB]$')
        plt.ticklabel_format(axis ='y',scilimits=(0,0))
        plt.xlabel('Delay [s]')
        plt.legend()
        plt.tight_layout()
        if save == 'True':
            plt.savefig('Pictures/DPI_Log_G_cl{}_G_r{}_w{}.png'.format(self.gamma_cl,self.gamma_ray,window),dpi = 500,bbox_inches='tight')
    
    def delay_power_intensity(self):
        
        t = np.linspace(0,self.t_max,self.I)
        
        if self.gamma_cl == self.gamma_ray:
            P_t = self.Lambda_cl+self.Lambda_ray+self.Lambda_ray*self.Lambda_cl*t*\
            np.exp(-t/self.gamma_ray)
            
        if self.gamma_cl != self.gamma_ray:
            k_1 = self.Lambda_cl * ( 1 + self.Lambda_ray * ((self.gamma_cl * self.gamma_ray)\
                                                 /(self.gamma_cl - self.gamma_ray)))
           
            k_2 = self.Lambda_ray*  (1 - self.Lambda_cl *((self.gamma_cl * self.gamma_ray)\
                                                 /(self.gamma_cl - self.gamma_ray)))
           
            P_t= self.Q*(k_1 * np.exp(-t / self.gamma_cl)\
                 + k_2 * np.exp(-t / self.gamma_ray))
            P_t[0] += self.Q

        return P_t,t
    
    def plot_of_delays(self):
        plt.figure()
        for j in range(len(self.T)):
            plt.plot(self.T[j]+self.tau_list[j],np.repeat(-j,len(self.T[j]+self.tau_list[j])),\
                     'x',label = r'$T_{}+\tau$'.format(j))
            plt.yticks([])



