# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:05:47 2023

@author: Ben Muller
"""
# %% Imports
import numpy as np
import torch

# %% Kernel Class

class Kernel:
    
    def __init__(self, lengthscales=None, eps=1e-6, name=None):
        if isinstance(lengthscales, torch.Tensor):
            self.loglengthscales = lengthscales.log()
        else:
            self.loglengthscales = torch.tensor(np.log(lengthscales), requires_grad=True, dtype=torch.float32)
        self.eps = eps
        self.name = name
        
    def set_lengthscales(self, lengthscales):
        self.loglengthscales = torch.tensor(np.log(lengthscales), requires_grad=True, dtype=torch.float32)
    
    def get_lengthscales(self):
        return self.loglengthscales.exp()
    
    def get_loglengthscales(self):
        return self.loglengthscales
        
# %% Kernel functions

class RBFKernel(Kernel):

    def __init__(self, lengthscales, eps=1e-6):
        super().__init__(lengthscales, eps, "RBFKernel")
        
        
    def __call__(self, X1, X2):    
        X1 = X1 / self.loglengthscales.exp()
        X2 = X2 / self.loglengthscales.exp()
        X1_norm = torch.sum(X1 * X1, axis=1).view(-1,1)
        X2_norm = torch.sum(X2 * X2, axis=1).view(-1,1)
        K = X1_norm.expand(X1.size(0), X2.size(0)) + X2_norm.t().expand(X1.size(0), X2.size(0)) - 2.0 * X1 @ X2.t()
        return torch.exp(- 0.5 * K)
    
    def get_num_params(num_dim):
        return num_dim
    
class CompositeRBFKernel(Kernel):

    def __init__(self, lengthscales, eps=1e-6):
        super().__init__(lengthscales, eps)
        
        
    def __call__(self, X1, X2):    
        X1 = X1 * torch.sqrt(self.loglengthscales.exp())
        X2 = X2 * torch.sqrt(self.loglengthscales.exp())
        X1_norm = torch.sum(X1 * X1, axis=1).view(-1,1)
        X2_norm = torch.sum(X2 * X2, axis=1).view(-1,1)
        K = X1_norm.expand(X1.size(0), X2.size(0)) + X2_norm.t().expand(X1.size(0), X2.size(0)) - 2.0 * X1 @ X2.t()
        return torch.exp(-K)
    
    def get_num_params(num_dim):
        return num_dim
    
class Matern12(Kernel):

    def __init__(self, lengthscales, eps=1e-6):
        super().__init__(lengthscales, eps, "Matern12")
    
    def __call__(self, X1, X2):
        X1 = X1 / self.loglengthscales.exp()
        X2 = X2 / self.loglengthscales.exp()
        r = torch.cdist(X1, X2, p=2)
        return torch.exp(-r)
    
    def get_num_params(num_dim):
        return num_dim
    
class Matern32(Kernel):

    def __init__(self, lengthscales, eps=1e-6):
        super().__init__(lengthscales, eps, "Matern32")
    
    def __call__(self, X1, X2):
        X1 = X1 / self.loglengthscales.exp()
        X2 = X2 / self.loglengthscales.exp()
        r = torch.cdist(X1, X2, p=2)
        return (1 + np.sqrt(3) * r) * torch.exp(- np.sqrt(3) * r)
    
    def get_num_params(num_dim):
        return num_dim
    
class Matern52(Kernel):

    def __init__(self, lengthscales, eps=1e-6):
        super().__init__(lengthscales, eps, "Matern52")
    
    def __call__(self, X1, X2):
        X1 = X1 / self.loglengthscales.exp()
        X2 = X2 / self.loglengthscales.exp()
        r = torch.cdist(X1, X2, p=2)
        return (1 + np.sqrt(5) * r + (5 / 3) * torch.pow(r, 2)) * torch.exp(- np.sqrt(5) * r)
    
    def get_num_params(num_dim):
        return num_dim
