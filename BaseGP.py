# -*- coding: utf-8 -*-

"""
Created on Tue Oct 24 13:45:40 2023

@author: benmu
"""

# %% Imports and setting env
import numpy as np
import matplotlib.pyplot as plt
import torch

# %% Base Gaussian Process

class BaseGP:
    
    def __init__(self, mean=None, covariance=None, init_mean_params = None, init_covar_params = None, noise = 1e-4, ynorm = True, nugget = 1e-4):
        self.mean = mean
        self.kernel = covariance
        self.init_mean_params = init_mean_params
        self.init_covar_params = init_covar_params
                
        # Standartised values
        self.Xmin = None; self.Xmax = None
        self.Ymean = None; self.Ystd = None
        self.ynorm = ynorm
        
        self.X = None
        self.Y = None
        
        self.n_data = None
        self.n_input_params = None
        self.n_output_params = None
        
        self.noise = noise
        self.nugget = nugget
        
        self.K = None
        self.K_inv = None
    
    def initial_data_scale(self, X, Y, is_unit_length=True):
        # Scaling X
        self.Xmin = X.min(0).values
        self.Xmax = X.max(0).values
        XS = self.standardise_X(X, is_unit_length)

        # Normalising Y
        if self.ynorm:
            self.Ymean = Y.mean()
            self.Ystd = Y.std()
            YS = self.normalise_Y(Y)
        else:
            YS = Y
            
        self.X = XS
        self.Y = YS
            
        return XS, YS
    
    def standardise_X(self, X, is_unit_length=True):
        if is_unit_length:
            return (X - self.Xmin) / (self.Xmax - self.Xmin)
        else:
            return 2 * (X - self.Xmin) / (self.Xmax - self.Xmin) - 1
    
    def destandardise_X(self, X, is_unit_length=True):
        if is_unit_length:
            return X * (self.Xmax - self.Xmin) + self.Xmin
        else:
            return ((X + 1) / 2) * (self.Xmax - self.Xmin) + self.Xmin
    
    def normalise_Y(self, Y):
        return (Y - self.Ymean) / self.Ystd
    
    def denormalise_Y(self, Y):
        return Y * self.Ystd + self.Ymean
    
    def initialise_mean_covar(self):
        if self.init_mean_params is None:
            input_num = self.mean.get_num_params(self.n_input_params)
            if input_num == 0:
                self.mean = self.mean()
            elif input_num == 1:
                self.mean = self.mean([0.])
            else:
                self.mean = self.mean([0.] + [1. for _ in range(input_num - 1)])
        else:
            self.mean = self.mean(self.init_mean_params)
        if self.init_covar_params is None:
            self.kernel = self.kernel([1. for _ in range(self.kernel.get_num_params(self.n_input_params))]) 
        else:
            self.kernel = self.kernel(self.init_covar_params)
    
    def initialise_covar(self):
        if self.init_covar_params is None:
            self.kernel = self.kernel([1. for _ in range(self.kernel.get_num_params(self.n_input_params))]) 
        else:
            self.kernel = self.kernel(self.init_covar_params)
                
    
    def base_plot_1d(self, XX, mean, sd, X, Y, plot_mean=True, samples=None, plot_rows=1, plot_cols=1, plot_index=1, show_confint=True):
        fig = plt.figure()
        ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
        if plot_mean:
            ax.plot(XX, mean)
        if samples is not None:
            ax.plot(XX, samples)
        ax.plot(X, Y, 'o', color = 'navy')
        if show_confint:
            ax.fill_between(XX, mean + 2 * sd, mean - 2 * sd, alpha=0.4)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        return fig, ax
    
    def base_plot_2d(self, XX, mean, plot_rows=1, plot_cols=1, plot_index=1):
        fig = plt.figure()
        ax = fig.add_subplot(plot_rows, plot_cols, plot_index, projection = '3d')
        ax.plot_trisurf(XX[:,0], XX[:,1], mean, cmap=plt.cm.coolwarm, edgecolor='none')
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        return fig, ax
        