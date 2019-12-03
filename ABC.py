#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:58:52 2019

@author: Jonas
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import elfi
from skimage.feature import peak_local_max
from scipy.stats import gaussian_kde
import pickle as p
from Data_measurement_model import SPGP_meas
from sklearn.linear_model import LinearRegression
from scipy.stats import median_absolute_deviation
from scipy.spatial.distance import cdist

pi = np.pi
cos = np.cos
sin = np.sin

def kde_scipy(x,x_grid,bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
        
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

class ABC():
    """
    Perfroms ABC on the spencer model, but only for sigma
    """
    def __init__(self,N=2000,quantile = 0.05,n = 30,\
                 start_prior = 0.001,end_prior = 20, nr_peaks = 10,seed = False,\
                 sigma0 = 1,mode = 'SMC',distance = 'euclidean'):
        self.n = 30
        self.model = SPGP_meas(n = self.n)
        self.quantile = quantile
        self.N = N
        self.idx = self.search_area()
        self.start_prior = start_prior
        self.end_prior = end_prior
        self.nr_peaks = nr_peaks
        self.mode = mode
        if seed == True:
            np.random.seed(90)
        self.sigma0 = sigma0
        self.K = 800
        self.T = 7
        self.distance = distance
        self.mad = np.load('median_absoulte_deviation.npy')
    
    def Her(self,x): # hermitian
        return np.conj(x).T


    def e(self,theta):
        """
        imping wave direction
        """
        return np.array([cos(theta),sin(theta)])
    
    
    def steering_vec(self,theta):  
        """
        
        Note: g(theta,phi) = 1 by default (no callibration possible)
        """
        r=self.model.r
        lamb = self.model.wave_len
        e_vec = self.e(theta)
        a = np.exp((2j*pi/lamb)*np.dot(r, e_vec))
        return np.asarray(a)
    
    def Beamscan(self,H, theta_rng=[0, pi], res_t=801):
        """
        Calculated the Power Delay Azimuth Spectrum
        """
        
        H = H.reshape(self.model.n,self.model.I)
        P_bf = np.zeros((res_t,res_t),dtype = complex)
        theta = np.linspace(theta_rng[0], theta_rng[1], res_t)
        for j in range(len(theta)):
            a = self.steering_vec(theta[j])
            P_bf[j,:] = np.abs(self.Her(H).dot(a))
        return P_bf.real,theta
    
    def search_area(self,I = 801,percentage = 0.1):
        area = np.arange(I)
        start = int(400-(400*percentage))
        end = int(400+(400*percentage))
        return area[start:end]

    def var(self,H):
        """
        Variance of the Power delay azimuth spectrum
        """
        
        idx = self.idx
        P,ang = self.Beamscan(H)
        peaks = peak_local_max(P[idx,:self.K],min_distance=1,num_peaks=self.nr_peaks,\
                               exclude_border=False) # 10 virker fint
        peaks[:, 0] += idx[0]
        return np.var(ang[peaks[:,0]])
        
    def mean(self,H):
        """
        Mean of the Power delay Azimuth spectrum
        """
        
        idx = self.idx
        P,ang = self.Beamscan(H)
        peaks = peak_local_max(P[idx,:self.K],min_distance=1,num_peaks=self.nr_peaks\
                               ,exclude_border=False)
        peaks[:, 0] += idx[0]
        return np.mean(ang[peaks[:,0]])
    
    def max_min_diff(self,H):
        idx = self.idx
        P,ang = self.Beamscan(H)
        peaks = peak_local_max(P[idx,:self.K],min_distance=1,num_peaks=self.nr_peaks\
                               ,exclude_border=False)
        peaks[:, 0] += idx[0]
        return np.max(abs(ang[peaks[:,0]].max()-ang[peaks[:,0]].min()))
    
    def PDP_diff_mean(self,H):
        """
        Power delay profile difference
        """
        
        h = np.fft.ifft(H[[0,-1],:],axis = 1)
        diff = np.abs(h[0,:]-h[-1,:])**2
        return np.mean(diff)
    
    def PDP_diff_var(self,H):
        """
        Power delay profile difference
        """
        
        h = np.fft.ifft(H[[0,-1],:],axis = 1)
        diff = np.abs(h[0,:]-h[-1,:])**2
        return np.var(diff)
    
    def simulator(self,sigma,batch_size = 1,random_state = None):
        """
        Used to construct the elfi simulator
        """
        
        return self.model.simulate_meas(sig = sigma)
    
    def weighted_euclidean(self,*summaries, observed):

        s_s,o_s = summaries/self.mad,observed/self.mad
        
        return np.sum(np.abs(s_s - o_s))
    
    def Perfrom_ABC(self,save = False):
        
        sigma_prior = elfi.Prior('uniform',self.start_prior,self.end_prior)

        
        y0 = self.simulator(self.sigma0)
        sim = elfi.Simulator(self.simulator,sigma_prior,observed=y0)
        
        S1 = elfi.Summary(self.var,sim)
        S2 = elfi.Summary(self.max_min_diff,sim)
        S3 = elfi.Summary(self.PDP_diff_mean,sim)
        S4 = elfi.Summary(self.PDP_diff_var,sim)
        S5 = elfi.Summary(self.mean,sim)
        sumstats = []
        sumstats.append(S1)
        sumstats.append(S2)
        sumstats.append(S3)
        sumstats.append(S4)
        sumstats.append(S5)
        output_names = 'S1,S2,S3,S4,S5'
        output_list = ['S1','S2','S3','S4','S5']
        if self.distance == 'euclidean':
            d = elfi.Distance('euclidean',*sumstats)
                
        elif self.distance == 'seuclidean':
            d = elfi.Distance('seuclidean',*sumstats,V = None)
            
        if self.mode == 'SMC':
            rej_temp  = elfi.Rejection(d.model['d'],output_names = output_list)
            rej = elfi.SMC(d.model['d'],output_names = rej_temp.output_names)
            quantile_list = list(np.repeat(self.quantile,self.T))
            res_sample = rej.sample(self.N,quantile_list)
            try:
                adj_res = elfi.adjust_posterior(res_sample.T, d.model, output_list)
            except:
                print('Error in calculation of adjustment')
                adj_res = 'NaN'
        
        if self.mode == 'Rejection':
            rej = elfi.Rejection(d.model['d'],output_names = output_list)
            res_sample = rej.sample(self.N,self.quantile)
            try:
                adj_res = elfi.adjust_posterior(res_sample.T, d.model, output_list)
            except:
                print('Error in calculation of adjustment')
                adj_res = 'NaN'
        
        dat = res_sample.samples_array[:,-1]
        dat_grid = np.linspace(self.start_prior,self.end_prior,self.N)
        kde = kde_scipy(dat,dat_grid,bandwidth = 1)
        
        self.data_dict = {'theta':res_sample.samples_array[:,-1],\
                          'summariers':np.array([res_sample.outputs['S1'],res_sample.outputs['S2'],\
                        res_sample.outputs['S3'],res_sample.outputs['S4'],res_sample.outputs['S5']]),\
             'True sim': y0, 'true summaries' : np.array([self.var(y0),\
                            self.max_min_diff(y0),self.PDP_diff_mean(y0),self.PDP_diff_var(y0),self.mean(y0),]),\
             'kde' : kde, 'kde_grid':dat_grid,'sigma0': self.sigma0,\
             'elfi adjust' : adj_res,\
             'elfi results' : res_sample}
        if save == True:
            try:
                string = str(self.quantile)[-1]
                f = open('examples/Pickle_data_%s_q_%s_sig0_%d_summaries_%s_distance_%s'%(self.mode,string,int(self.sigma0),output_names,self.distance),'wb')
                p.dump(self.data_dict,f)
                f.close()
            except:
                print('Error in saving')
        return self.data_dict
        
    
    
# =============================================================================
# Example
# =============================================================================

ABC_test = ABC()
ABC_res = ABC_test.Perfrom_ABC(save = True)
