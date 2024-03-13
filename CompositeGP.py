# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:59:14 2024

@author: benmu
"""

# %% Imports and setting env
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
from BaseGP import BaseGP

from Kernels import CompositeRBFKernel
        
# %% Gaussian Process
    
class CompositeGP(BaseGP):
    
    def __init__(self, noise = 1e-4, ynorm = True, optim_itter = 50, lr = 1):
        super().__init__(None, None, None, None, noise, ynorm)
        self.kernels = None
        self.optim_itter = optim_itter
        self.lr = lr
        
        # parameters
        self.alpha = None
        self.kappa = None
        self.b = torch.tensor(0.5, requires_grad=True)
        self.lamb = torch.tensor(0.5, requires_grad=True)
        self.sigma = None
        self.sigma12 = None
        
        
    def fit(self, X, Y):
        # Initialising data
        self.n_data, self.n_input_params = X.size()
        self.n_output_params = Y.size(1)
        XS, YS = self.initial_data_scale(X, Y)
        
        # Initial values
        self.alpha = self.calc_alpha()
        self.kappa = self.alpha.clone().detach().requires_grad_(True)
        
        # Initial sigma is nxn identity matrix
        self.sigma = torch.eye(self.n_data)
        self.sigma12 = torch.eye(self.n_data)
        
        # Initialising Covariances
        kernel_parameters = [5. for _ in range(self.n_input_params)]
        self.kernels = [CompositeRBFKernel(kernel_parameters), None]
        
        # Optimise GP
        self.optimise()
        
        
    def predict(self, XX):
        # Scale data
        XXS = self.standardise_X(XX)
        
        # Useful vector in calculations
        ones = torch.ones((self.n_data, 1))
        
        # Calculate kernels
        G = self.kernels[0](self.X, self.X)
        GX = self.kernels[0](self.X, XXS)
        L = self.kernels[1](self.X, self.X)
        LX = self.kernels[1](self.X, XXS)
        
        # Calculate q and Q
        Q = G + self.lamb * self.sigma12 @ L @ self.sigma12
        Qinv = torch.linalg.inv(Q)
        gb = self.gb(GX.T, self.b)
        v = self.v(gb, self.s2)
        q = GX + self.lamb * torch.sqrt(v).T * (self.sigma12 @ LX) 
        
        # Calculate Mean
        mu = self.mu(G, self.lamb, self.sigma12, L, self.Y)
        mean = mu + q.T @ Qinv @ (self.Y - mu) 
        
        # calculate Variance
        tau2 = self.tau2(G, self.lamb, self.sigma12, L, self.Y)
        var = tau2 * (1 + self.lamb * v - q.T @ Qinv @ q + (1 - q.T @ Qinv @ ones) % (1 - q.T @ Qinv @ ones).T  / (ones.T @ Qinv @ ones))
        
        # temp
        # mean = mu + GX.T @ torch.linalg.inv(G + self.lamb * self.sigma12 @ L @ self.sigma12) @ (self.Y - mu * ones)
        # mean += self.lamb * torch.sqrt(v) * (LX.T @ self.sigma12) @ Qinv @ (self.Y - mu)
        
        # Rescale mean and variance
        if self.ynorm:
            mean = self.denormalise_Y(mean)
            var = var * torch.pow(self.Ystd, 2)
            
        return mean, var
        
        
    def optimise(self):
        # Get parameters
        parameters = [self.kernels[0].get_loglengthscales(), self.kappa, self.b, self.lamb]
        
        # Initialise optimiser
        optimiser = torch.optim.Adam(parameters, lr=self.lr)
        
        # Do training
        for i in range(self.optim_itter):
            optimiser.zero_grad()
            loss = self.negative_log_likelihood()
            loss.backward()
            # print(i, ":", round(float(loss.detach().numpy()), 4))
            optimiser.step()
            
            # Restrict the parameter values
            with torch.no_grad():
                parameters[0].data = torch.clamp(parameters[0].data, -torch.inf, torch.log(self.alpha))
                parameters[1].data = torch.clamp(parameters[1].data, self.alpha, torch.inf)
                parameters[2].data = torch.clamp(parameters[2].data, 0, 1)
                parameters[3].data = torch.clamp(parameters[3].data, 1e-4, 1)
                        
        # Calculate the Kernels to find the sigmas
        G = self.kernels[0](self.X, self.X)
        L = CompositeRBFKernel(self.kernels[0].get_lengthscales() + self.kappa)(self.X, self.X)
        sigma12 = torch.eye(self.n_data)
        
        # Itteratively update the Sigmas
        for i in range(4):
            s2 = self.calc_s2(self.Y, G, self.lamb, L, sigma12)
            sigma = self.calc_sigma(G, self.b, s2)
            sigma12 = torch.sqrt(sigma)
            
        # Save sigma and s^2 values
        self.s2 = s2
        self.sigma = sigma
        self.sigma12 = sigma12
        
        # Store second kernel (local kernel)
        self.kernels[1] = CompositeRBFKernel(self.kernels[0].get_lengthscales() + self.kappa)
        
    def negative_log_likelihood(self):
        # Calculate Kernels
        G = self.kernels[0](self.X, self.X)
        L = CompositeRBFKernel(self.kernels[0].get_lengthscales() + self.kappa)(self.X, self.X)

        # Calcualte sigmas
        with torch.no_grad():
            sigma12 = torch.eye(self.n_data)
            for i in range(4):
                s2 = self.calc_s2(self.Y, G, self.lamb, L, sigma12)
                sigma = self.calc_sigma(G, self.b, s2)
                sigma12 = torch.sqrt(sigma)
            
        s2 = self.calc_s2(self.Y, G, self.lamb, L, sigma12)
        sigma = self.calc_sigma(G, self.b, s2)
        sigma12 = torch.sqrt(sigma)
        
        # Calculate mu and tau
        mu = self.mu(G, self.lamb, sigma12, L, self.Y)
        tau2 = self.tau2(G, self.lamb, sigma12, L, self.Y, mu)
        
        # print(tau2, mu,  torch.logdet(G + self.lamb * sigma12 @ L @ sigma12))        
                
        # return nll
        return self.n_data * torch.log(tau2) + torch.logdet(G + self.lamb * sigma12 @ L @ sigma12)
          
    def gb(self, G, b):
        # Calculate gb for variance calculations
        return G ** b
    
    def v(self, gb, s2):
        # calculate variance
        vx = (gb @ s2) / (gb @ torch.ones((self.n_data, 1))) 
        
        # standardise the variance
        return vx / torch.mean(vx)
    
    def calc_sigma(self, G, b, s2):
        # calculate standardised sigmas using the standardised variance
        gb = self.gb(G, b)
        vs = self.v(gb, s2).T[0]
        return torch.diag(vs)
    
    def calc_s2(self, Y, G, lamb, L, sigma12):
        ones = torch.ones((self.n_data, 1))
        mu = self.mu(G, lamb, sigma12, L, Y) 
        yglobal = mu + G.T @ torch.linalg.inv(G + lamb * sigma12 @ L @ sigma12) @ (Y - mu * ones)
        return torch.linalg.inv(sigma12) @ (Y - yglobal) ** 2
    
    def mu(self, G, lamb, sigma12, L, Y, GlambLsiginv=None):
        ones = torch.ones((self.n_data, 1))
        if GlambLsiginv is None:
            GlambLsiginv = torch.linalg.inv(G + lamb * sigma12 @ L @ sigma12) 
        return torch.linalg.inv(ones.T @ GlambLsiginv @ ones) @ (ones.T @ GlambLsiginv @ Y)
            
    def tau2(self, G, lamb, sigma12, L, Y, mu=None):
        ones = torch.ones((self.n_data, 1))
        GlambLsiginv = torch.linalg.inv(G + lamb * sigma12 @ L @ sigma12)
        if mu is None:
            mu = self.mu(G, lamb, sigma12, L, Y, GlambLsiginv)
        return (1 / self.n_data) * (Y - mu * ones).T @ GlambLsiginv @ (Y - mu * ones)
        
    def calc_alpha(self):
        return torch.log(torch.tensor(100)) / self.davg2()
    
    def davg2(self):
        X_norm = torch.sum(self.X * self.X, axis=1).view(-1,1)
        distance2 = X_norm.expand(self.n_data, self.n_data) + X_norm.t().expand(self.n_data, self.n_data) - 2.0 * self.X @ self.X.t()
        total_distance = 0
        for i in range(self.n_data - 1):
            for j in range(i + 1, self.n_data):
                total_distance += 1 / distance2[i,j]
        return (self.n_data * (self.n_data - 1)) / (2 * total_distance)

    def plot_1d(self, XX, plot_rows=1, plot_cols=1, plot_index=1):
        mean, var = self.predict(XX)
        mean = mean.T.detach().numpy()[0]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')            
            sd = np.sqrt(torch.diagonal(var).detach().numpy())
        XX = XX.T.detach().numpy()[0]
        X = self.destandardise_X(self.X)
        if self.ynorm:
            Y = self.denormalise_Y(self.Y)
        fig, ax = self.base_plot_1d(XX, mean, sd, X, Y, plot_rows=plot_rows, plot_cols=plot_cols, plot_index=plot_index)
        return fig, ax
    
    def plot_2d(self, XX, plot_rows=1, plot_cols=1, plot_index=1):
        mean, _ = self.predict(XX)
        mean = mean.t().detach().numpy()[0]
        x = XX.detach().numpy()
        fig, ax = self.base_plot_2d(x, mean, plot_rows=plot_rows, plot_cols=plot_cols, plot_index=plot_index)
        return fig, ax
    
    