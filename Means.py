# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:21:21 2023

@author: Ben Muller
"""
# %% Imports
import torch

# %% Mean class

class Means:
    def __init__(self, lengthscales=None, name=None):
        if isinstance(lengthscales, torch.Tensor):
            self.lengthscales = lengthscales
        else:
            self.lengthscales = torch.tensor(lengthscales, requires_grad=True)
        self.name = name
    
    def set_lengthscales(self, lengthscales):
        self.lengthscales = torch.tensor(lengthscales, requires_grad=True)
    
    def get_lengthscales(self):
        return self.lengthscales
    
        

# %% Mean funcitons

class ZeroMean(Means):
    def __init__(self):
        super().__init__(name="ZeroMean")
        
    def __call__(self, X):
        return torch.zeros(X.size(0),1)
    
    def get_num_params(num_dim):
        return 0

class ConstantMean(Means):
    def __init__(self, lengthscales):
        super().__init__(lengthscales, name="ConstantMean")
        
    def __call__(self, X):
        return self.lengthscales * torch.ones(X.size(0),1)
    
    def get_num_params(num_dim):
        return 1
    
class LinearMean(Means):
    def __init__(self, lengthscales):
        super().__init__(lengthscales, name="LinearMean")
        
    def __call__(self, X):
        return self.lengthscales[0] * torch.ones(X.size(0),1) + X @ self.lengthscales[1:].view(-1,1)
    
    def get_num_params(num_dim):
        return 1 + num_dim
    
class QuadraticMean(Means):
    def __init__(self, lengthscales, name="QuadraticMean"):
        super().__init__(lengthscales)
        
    def __call__(self, X):
        n_params = X.size(1)
        return self.lengthscales[0] * torch.ones(X.size(0),1) + X @ self.lengthscales[1:1 + n_params].view(-1,1) + torch.pow(X, 2) @ self.lengthscales[1+n_params:].view(-1,1)
    
    def get_num_params(num_dim):
        return 1 + 2 * num_dim
    
    