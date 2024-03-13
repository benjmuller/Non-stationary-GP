# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:47:17 2023

@author: benmu
"""

# %% Imports and setting env
import numpy as np
import matplotlib.pyplot as plt
import torch
import jax.numpy as jnp

import jax.random as jrand 
import jax.numpy as jnp
import numpyro
from numpyro.infer import NUTS, MCMC, HMC
from numpyro import sample
import numpyro.distributions as dist
from numpyro.diagnostics import summary

from BaseGP import BaseGP
import JaxMeans as jm
import JaxKernels as jk


        
# %% Gaussian Process
    
class GaussianProcess(BaseGP):
    
    def __init__(self, mean, covariance, init_mean_params=None, init_covar_params=None, noise = 1e-4, ynorm = True, nugget = 1e-4, mean_fit = True, optim_itter = 50, lr = 0.1):
        super().__init__(mean, covariance, init_mean_params, init_covar_params, noise, ynorm, nugget)
        self.mean_fit = mean_fit
        self.optimiser = None
        self.optim_itter = optim_itter
        self.lr = lr

    def fit(self, X, Y, optimise_param = True, optimiser='ADAM'):
        """Fit the gaussian Process"""
        # Data Preprocessing
        self.optimiser = optimiser
        self.n_data, self.n_input_params = X.size()
        self.n_output_params = Y.size(1)
        XS, YS = self.initial_data_scale(X, Y)
                
        # Initialise the mean and covariance
        self.initialise_mean_covar()
                
        # Optimise the parameters
        if optimise_param:
            if self.optimiser == 'ADAM':    
                self.ADAM_optimise()
            elif self.optimiser == 'LOOCV':
                self.LOOCV_optimise()
            elif self.optimiser == "Bayes":
                self.Bayes_optimise()
            else:
                raise Exception("Incorrect Optimiser")
             
        # Save kernel calculations
        self.K = self.kernel(XS, XS) + torch.eye(XS.size(0)) * (self.noise + self.nugget)
        self.K_inv = torch.linalg.inv(self.K) 
        
        
    def predict(self, XX):   
        """Predicts the GP"""
        XXS = self.standardise_X(XX)
        K_s = self.kernel(XXS, self.X)
        K_ss = self.kernel(XXS, XXS) + torch.eye(XXS.size(0)) * (self.noise + self.nugget)
        mean = self.mean(XXS) + K_s @ self.K_inv @ (self.Y - self.mean(self.X))
        var = K_ss - K_s @ self.K_inv @ K_s.T
        if self.ynorm:
            mean = self.denormalise_Y(mean)
            var = var * torch.pow(self.Ystd, 2)
    
        return mean, var
    
    
    def ADAM_optimise(self):
        """Implements Adam Optimisation for nll"""
        if self.mean_fit:
            parameters = [self.kernel.get_loglengthscales(), self.mean.get_lengthscales()]
        else:
            parameters = [self.kernel.get_loglengthscales()]
        optimiser = torch.optim.Adam(parameters, lr=self.lr)
        for i in range(self.optim_itter):
            optimiser.zero_grad()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimiser.step()
            
            
    def LOOCV_optimise(self):
        """Implements Adam optimsation for LOOCV"""
        if self.mean_fit:
            parameters = [self.kernel.get_loglengthscales(), self.mean.get_lengthscales()]
        else:
            parameters = [self.kernel.get_loglengthscales()]
            
        optimiser = torch.optim.Adam(parameters, lr=self.lr)
        for i in range(self.optim_itter):  
            optimiser.zero_grad()
            loss = 0
            for j in range(self.n_data):
                xpred, xvar = self.LOOCV_loss(j)
                loss += (xpred[0] - self.Y[j,:]) ** 2
                # if xvar < 1e-4:
                #     loss += (xpred[0] - self.Y[j,:]) ** 2 / torch.tensor([[1e-4]])
                # else:
                #     loss += (xpred[0] - self.Y[j,:]) ** 2 / xvar.sqrt()
            loss = loss / self.n_data
            loss.backward()
            optimiser.step()
        
            
    def Bayes_optimise(self):
        rng_key, _ = jrand.split(jrand.PRNGKey(1))
        
        X = jnp.asarray(self.X.detach().numpy())
        Y = jnp.asarray(self.Y.detach().numpy())
        
        # mean and kernel lookup table
        mean_table = {
                        "ZeroMean": jm.ZeroMean,
                        "ConstantMean": jm.ConstantMean,
                        "LinearMean": jm.LinearMean,
                        "QuadraticMean": jm.QuadraticMean
                    }
        
        kernel_table = {
                        "RBFKernel": jk.RBFKernel,
                        "Matern12": jk.Matern12,
                        "Matern32": jk.Matern32,
                        "Matern52": jk.Matern52
                        }
        
        mean = mean_table[self.mean.name](self.mean.get_lengthscales().detach().numpy())
        kernel = kernel_table[self.kernel.name](self.kernel.get_lengthscales().detach().numpy())
        
        HMC_GPkernel = NUTS(self.HMCGPModel, max_tree_depth=5, dense_mass=False)
        mcmc = MCMC(
            HMC_GPkernel,
            num_warmup=250,
            num_samples=250
        )
        mcmc.run(rng_key, X, Y, mean, kernel)
        
        # mcmc.print_summary()
        
        posterior_samples = mcmc.get_samples()
        summary_dict = summary(posterior_samples, group_by_chain=False)
        betas = summary_dict["betas"]["mean"]
        deltas = summary_dict["deltas"]["mean"]
        sigma2 = float(summary_dict["sigma2"]["mean"])
        self.mean = type(self.mean)(betas)
        self.kernel = type(self.kernel)(deltas)
        self.noise = sigma2
        
        
    def HMCGPModel(self, X, Y, mean, kernel):
        betas =  sample("betas", dist.Normal(loc=jnp.zeros(len(mean.get_lengthscales())), scale=jnp.sqrt(10)))
        deltas = sample("deltas", dist.Gamma(4, jnp.ones(len(kernel.get_lengthscales())) * 4))
        sigma2 = sample("sigma2", dist.InverseGamma(jnp.ones(1) * 2, 1))
        mean, kernel = self.calc_mean_kernel(X, Y, deltas, betas, sigma2, mean, kernel)
        sample("obs", dist.MultivariateNormal(loc=mean, covariance_matrix=kernel), obs=Y.T[0])
        
        
    def calc_mean_kernel(self, X, Y, deltas, betas, sigma2, mean, kernel):
        mu = type(mean)(betas)(X)
        kern = type(kernel)(deltas)(X, X) + jnp.eye(self.n_data) * sigma2
        return mu, kern

            
    def LOOCV_loss(self, j):
        """Calculates the predictions for the LOOCV for data-point j"""
        # Getting GP parameters
        mean = type(self.mean)
        mean_param = self.mean.get_lengthscales()
        kern = type(self.kernel)
        kern_param = self.kernel.get_loglengthscales().exp()
        
        # Splitting into training data
        indices = torch.arange(self.n_data)
        indices = indices[indices != j]
        X_train = self.X[indices,:]
        Y_train = self.Y[indices,:]
                
        GP = GaussianProcess(mean, kern, mean_param, kern_param)
        GP.fit(X_train, Y_train, optimise_param=False)
        xpred, xvar = GP.predict(self.X[j,:].view((1, self.n_input_params)))
        return xpred, xvar
                
        
    def negative_log_likelihood(self):
        """Calculates the nll"""
        K = self.kernel(self.X, self.X)
        chol = torch.linalg.cholesky(K + torch.eye(self.n_data) * self.noise)
        gamma = torch.linalg.solve_triangular(chol, self.Y - self.mean(self.X), upper = False)
        data_fit = 0.5 * torch.pow(gamma, 2).sum()
        complexity = chol.diag().log().sum()
        normalising = (self.n_data / 2) * torch.log(torch.tensor([2 * torch.pi]))
        return data_fit + complexity + normalising
    
    
    def sample_posterior(self, XX, n_samples = 10, return_posterior = False):
        """Returns samples from the posterior distribution"""
        mean, covar = self.predict(XX)
        mean = mean.t()[0].detach().numpy()
        covar = covar.detach().numpy()
        sampler = np.random.multivariate_normal(mean, covar, size=n_samples, check_valid='ignore')
        if return_posterior:
            return mean, covar, sampler.transpose()
        return sampler.transpose()
    
    
    def plot_1d(self, XX, plot_rows=1, plot_cols=1, plot_index=1):
        mean, covar = self.predict(XX)
        mean = mean.t().detach().numpy()[0]
        sd = torch.sqrt(torch.diagonal(covar)).detach().numpy()
        x = XX.t().detach().numpy()[0]
        X = self.destandardise_X(self.X).detach().numpy()
        if self.ynorm:
            Y = self.denormalise_Y(self.Y).detach().numpy()
        else:
            Y = self.Y.detach().numpy()

        fig, ax = self.base_plot_1d(x, mean, sd, X, Y, plot_rows=plot_rows, plot_cols=plot_cols, plot_index=plot_index)
        return fig, ax
    
    
    def plot_1d_samples(self, XX, n_samples = 10, plot_mean=False, plot_rows=1, plot_cols=1, plot_index=1, show_confint=False):
        mean, covar, samples = self.sample_posterior(XX, n_samples = n_samples, return_posterior=True)
        sd = torch.sqrt(torch.diagonal(torch.from_numpy(covar))).detach().numpy()
        x = XX.t().detach().numpy()[0]
        X = self.destandardise_X(self.X).detach().numpy()
        if self.ynorm:
            Y = self.denormalise_Y(self.Y).detach().numpy()
        else:
            Y = self.Y.detach().numpy()

        fig, ax = self.base_plot_1d(x, mean, sd, X, Y, plot_mean=plot_mean, samples=samples, plot_rows=plot_rows, plot_cols=plot_cols, plot_index=plot_index, show_confint=show_confint)
        return fig, ax
    
    
    def plot_2d(self, XX, plot_rows=1, plot_cols=1, plot_index=1):
        mean, _ = self.predict(XX)
        mean = mean.t().detach().numpy()[0]
        x = XX.detach().numpy()
        fig, ax = self.base_plot_2d(x, mean, plot_rows=plot_rows, plot_cols=plot_cols, plot_index=plot_index)
        return fig, ax
        
        