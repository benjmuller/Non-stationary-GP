# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:57:52 2023

@author: benmu
"""

# %% Imports
import numpy as np
import jax.numpy as jnp
import jax

# %% Kernel Class

class JaxKernel:
    
    def __init__(self, lengthscales=None, eps=1e-6, name=None):
        if isinstance(lengthscales, jax.Array):
            self.loglengthscales = jnp.log(lengthscales)
        else:
            self.loglengthscales = jnp.log(jnp.asarray(lengthscales))
        self.eps = eps
        self.name = name
        
    def set_lengthscales(self, lengthscales):
        self.loglengthscales = jnp.log(jnp.asarray(lengthscales))
    
    def get_lengthscales(self):
        return jnp.exp(self.loglengthscales)
    
    def get_loglengthscales(self):
        return self.loglengthscales
        
# %% Kernel functions

class RBFKernel(JaxKernel):

    def __init__(self, lengthscales, eps=1e-6):
        super().__init__(lengthscales, eps, "JaxRBFKernel")
        
        
    def __call__(self, X1, X2):    
        X1 = X1 / jnp.exp(self.loglengthscales)
        X2 = X2 / jnp.exp(self.loglengthscales)
        X1_norm = jnp.sum(X1 * X1, axis=1).reshape(-1,1)
        X2_norm = jnp.sum(X2 * X2, axis=1).reshape(-1,1)
        K = jnp.broadcast_to(X1_norm, (X1.shape[0], X2.shape[0])) + jnp.broadcast_to(X2_norm.T, (X1.shape[0], X2.shape[0])) - 2.0 * X1 @ X2.T
        return jnp.exp(- 0.5 * K)
    
    def get_num_params(num_dim):
        return num_dim
    
class Matern12(JaxKernel):

    def __init__(self, lengthscales, eps=1e-6):
        super().__init__(lengthscales, eps, "JaxMatern12")
    
    def __call__(self, X1, X2):
        X1 = X1 / jnp.exp(self.loglengthscales)
        X2 = X2 / jnp.exp(self.loglengthscales)
        r = jax.linalg.norm(X1 - X2, 2)
        return jnp.exp(-r)
    
    def get_num_params(num_dim):
        return num_dim
    
class Matern32(JaxKernel):

    def __init__(self, lengthscales, eps=1e-6):
        super().__init__(lengthscales, eps, "JaxMatern32")
    
    def __call__(self, X1, X2):
        X1 = X1 / jnp.exp(self.loglengthscales)
        X2 = X2 / jnp.exp(self.loglengthscales)
        r = jax.linalg.norm(X1 - X2, 2)
        return (1 + jnp.sqrt(3) * r) * jnp.exp(- jnp.sqrt(3) * r)
    
    def get_num_params(num_dim):
        return num_dim
    
class Matern52(JaxKernel):

    def __init__(self, lengthscales, eps=1e-6):
        super().__init__(lengthscales, eps, "JaxMatern52")
    
    def __call__(self, X1, X2):
        X1 = X1 / jnp.exp(self.loglengthscales)
        X2 = X2 / jnp.exp(self.loglengthscales)
        r = jax.linalg.norm(X1 - X2, 2)
        return (1 + jnp.sqrt(5) * r + (5 / 3) * jnp.pow(r, 2)) * jnp.exp(- jnp.sqrt(5) * r)
    
    def get_num_params(num_dim):
        return num_dim
    
