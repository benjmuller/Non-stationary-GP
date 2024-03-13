# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:48:36 2023

@author: benmu
"""

# %% Imports and setting env
import numpy as np
import torch
import matplotlib.pyplot as plt

from BaseGP import BaseGP
from GaussianProcess import GaussianProcess
from Tree import Tree
from RJMCMC import TreedRJMCMC
       
        

# %% Treed GP

class TreedGP(BaseGP):
    
    def __init__(self, ynorm = True, n_itter = 0, n_burnin = 1000):
        super().__init__(None, None, None, None, None, ynorm, None)
        self.RJMCMC = None
        self.n_itter = n_itter
        self.n_burnin = n_burnin
        self.tree = None
        
        
    def fit(self, X, Y):
        self.X, self.Y = self.initial_data_scale(X, Y)
        self.n_data, self.n_input_params = X.size()
        self.n_output_params = Y.size(1)
        self.RJMCMC = self.optimise()
    
    def predict(self, XX):
        XXS = self.standardise_X(XX)
        mean, variance = self.RJMCMC.predict(XXS)
        mean = self.denormalise_Y(mean)
        variance = variance * torch.pow(self.Ystd, 2)
        return mean, variance
    
    def optimise(self):
        RJMCMC = TreedRJMCMC(self.X, self.Y, self.n_itter, self.n_burnin, self.Xmin, self.Xmax)
        RJMCMC.forward()
        self.tree = RJMCMC.trees
        return RJMCMC
        
    
    def plot_1d(self, XX, plot_rows=1, plot_cols=1, plot_index=1):
        mean, var = self.predict(XX)
        mean = mean.t().detach().numpy()[0]
        sd = torch.sqrt(var.t()).detach().numpy()[0]
        # print(mean, sd)
        x = XX.t().detach().numpy()[0]
        X = self.destandardise_X(self.X).t().detach().numpy()[0]
        Y = self.denormalise_Y(self.Y).t().detach().numpy()[0]
        fig, ax = self.base_plot_1d(x, mean, sd, X, Y, plot_rows=plot_rows, plot_cols=plot_cols, plot_index=plot_index)
        if self.tree.size > 0:
            locs = self.tree[0].get_splits()
            ymin = np.min(Y)
            ymax = np.max(Y)
            for _, xval in locs:
                xval = self.destandardise_X(xval).detach().numpy()
                ax.plot([xval, xval], [ymin, ymax], "--r")
        return fig, ax
    
    
    def plot_2d(self, XX, plot_rows=1, plot_cols=1, plot_index=1):
        mean, _ = self.predict(XX)
        mean = mean.t().detach().numpy()[0]
        x = XX.detach().numpy()
        fig, ax = self.base_plot_2d(x, mean, plot_rows=plot_rows, plot_cols=plot_cols, plot_index=plot_index)
        return fig, ax
            
