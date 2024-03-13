# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:03:02 2023

@author: benmu
"""

# %% Imports
import jax.numpy as jnp
import jax

# %% Mean class

class JaxMeans:
    def __init__(self, lengthscales=None, name=None):
        if isinstance(lengthscales, jax.Array):
            self.lengthscales = lengthscales
        else:
            self.lengthscales = jnp.asarray(lengthscales)
        self.name = name
    
    def set_lengthscales(self, lengthscales):
        self.lengthscales = jnp.asarray(lengthscales)
    
    def get_lengthscales(self):
        return self.lengthscales
    
        

# %% Mean funcitons

class ZeroMean(JaxMeans):
    def __init__(self):
        super().__init__(name="ZeroMean")
        
    def __call__(self, X):
        return jnp.zeros((X.shape[0],1))
    
    def get_num_params(num_dim):
        return 0

class ConstantMean(JaxMeans):
    def __init__(self, lengthscales):
        super().__init__(lengthscales, "JaxConstantMean")
        
    def __call__(self, X):
        return self.lengthscales * jnp.ones((X.shape[0],1))
    
    def get_num_params(num_dim):
        return 1
    
class LinearMean(JaxMeans):
    def __init__(self, lengthscales):
        super().__init__(lengthscales, "JaxLinearMean")
        
    def __call__(self, X):
        return self.lengthscales[0] * jnp.ones((X.shape[0],1)) + jnp.sum(self.lengthscales[1:] * X, axis=1).reshape(-1,1)
    
    def get_num_params(num_dim):
        return 1 + num_dim
    
class QuadraticMean(JaxMeans):
    def __init__(self, lengthscales):
        super().__init__(lengthscales, "JaxQuadraticMean")
        
    def __call__(self, X):
        n_params = X.shape[1]
        return self.lengthscales[0] * jnp.ones((X.shape[0],1)) + jnp.sum(self.lengthscales[1:1 + n_params] * X + self.lengthscales[1+n_params:] * (X ** 2), axis=1).reshape(-1,1)
    
    def get_num_params(num_dim):
        return 1 + 2 * num_dim
    
    
if __name__ == "__main__":
    x = jnp.array([[1, 1],[2, 2],[3,3],[4,4],[5,5]])
    mean = LinearMean([1, 2, 3])
    print(mean(x))
    