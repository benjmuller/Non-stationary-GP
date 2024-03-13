# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:18:04 2023

@author: benmu
"""

# %% Imports
import numpy as np
import torch

import jax
import jax.random as jrand 
import jax.numpy as jnp
from jax.lax import broadcast_shapes
import numpyro
from numpyro.infer import HMC, NUTS, MCMC, MixedHMC
from numpyro import sample, deterministic
import numpyro.distributions as dist
from numpyro.distributions import transforms
from numpyro.diagnostics import summary
from arviz import waic, from_numpyro
import warnings


from BaseGP import BaseGP
from GaussianProcess import GaussianProcess
from Means import ZeroMean, LinearMean
from Kernels import RBFKernel
import JaxMeans as jm



# %% Kernel Mixture GP

class KernelMixtureGP(BaseGP):
    
    def __init__(self, covariances, num_mixtures = 4, noise = 1e-4, ynorm = True, nugget = 1e-4):
        super().__init__(mean=jm.LinearMean, covariance=covariances, noise=noise, ynorm=ynorm, nugget=nugget)
        self.num_mixtures = num_mixtures
        self.alphas = None
        self.zetas = None
        self.deltas = None
        self.betas = None
        self.X_equal = None
        
    def fit(self, X, Y):
        """Fits the Gaussian Process"""
        self.n_data, self.n_input_params = X.size()
        self.n_output_params = Y.size(1)
        XS, YS = self.initial_data_scale(X, Y, is_unit_length=False)
        self.optimise()
        
    def predict(self, XX):
        """Predicts valuse of the Kernel Mixture GP"""
        # Data Preprocessing
        XX_n_data = XX.size(0)
        XXS = self.standardise_X(XX, is_unit_length=False)
        XX = jnp.asarray(XXS.detach().numpy())
        X = jnp.asarray(self.X.detach().numpy())
        Y = jnp.asarray(self.Y.detach().numpy())
        
        # Finds coordinates that are the same to find for the z calculations
        X_equal = jnp.broadcast_to(X, (X.shape[0], X.shape[0], self.n_input_params))
        XT_equal = jnp.moveaxis(X_equal, 0, 1)
        XMix_equal = jnp.broadcast_to(X, (XX.shape[0], X.shape[0], self.n_input_params))
        XMix_equal = jnp.moveaxis(XMix_equal, 0, 1)
        XMixT_equal = jnp.broadcast_to(XX, (X.shape[0], XX.shape[0], self.n_input_params))
        XX_equal = jnp.broadcast_to(XX, (XX.shape[0], XX.shape[0], self.n_input_params))
        XXT_equal = jnp.moveaxis(XX_equal, 0, 1)
        
        #Find the nuggets
        X_nugget = self.jax_nugget_z(X, self.alphas, self.num_mixtures)
        XX_nugget = self.jax_nugget_z(XX, self.alphas, self.num_mixtures)
               
        # Calculate the z's
        K = self.nugget * X_nugget @ X_nugget.T * jnp.equal(X_equal, XT_equal).all(axis=2) # jnp.equal(X, X.T)
        K_s = self.nugget * X_nugget @ XX_nugget.T * jnp.equal(XMix_equal, XMixT_equal).all(axis=2) # jnp.equal(X, XX.T)
        K_ss = self.nugget * XX_nugget @ XX_nugget.T * jnp.equal(XX_equal, XXT_equal).all(axis=2) # jnp.equal(XX, XX.T)
        
        # Calculate the lambdas
        X_lambdas = self.jax_calc_lambdas(X, self.alphas, self.num_mixtures)
        XS_lambdas = self.jax_calc_lambdas(XX, self.alphas, self.num_mixtures)
        
        # Calculate the kernel mixtures contributions to the kernels
        for i in range(self.num_mixtures):
            kern = type(self.kernel)(self.deltas[i])
            K += X_lambdas[:,i].reshape((self.n_data, 1)) @ (X_lambdas[:,i]).reshape((1, self.n_data)) * kern(X, X) + jnp.eye(X.shape[0]) * self.noise[i]
            K_s += X_lambdas[:,i].reshape((self.n_data, 1)) @ XS_lambdas[:,i].reshape((1, XX_n_data)) * kern(X, XX) + jnp.equal(XMix_equal, XMixT_equal).all(axis=2) * self.noise[i]
            K_ss += XS_lambdas[:,i].reshape((XX_n_data, 1)) @ XS_lambdas[:,i].reshape((1, XX_n_data)) * kern(XX, XX) + jnp.eye(XX.shape[0]) * self.noise[i]
        
        # Calculate the inverse of the kernel
        K_inv = jnp.linalg.inv(K)
        
        # Calculate the predicted values
        mean = self.mean(XX) + K_s.T @ K_inv @ (Y - self.mean(X))
        var = K_ss - K_s.T @ K_inv @ K_s
        
        # Data Postprocessing
        mean = torch.from_numpy(np.asarray(mean))
        var = torch.from_numpy(np.asarray(var))
        
        if self.ynorm:
            mean = self.denormalise_Y(mean)
            var = var * torch.pow(self.Ystd, 2)
    
        return mean, var
    
    def optimise(self):
        """Optimises the parameters for the Kernel mixture and GP"""
        # Collect data and get residuals
        residuals = self.LOO_residuals()
        residuals = jnp.asarray(residuals)
        X = jnp.asarray(self.X.detach().numpy())
        Y = jnp.asarray(self.Y.detach().numpy())
        
        # Calcualting X_equal
        self.get_X_equal(X)
                              
        # Getting random key for MCMC
        rng_key, _ = jrand.split(jrand.PRNGKey(1))
        
        # Fitting Mixtures
        self.fit_mixtures(X, Y, residuals, rng_key)
        
        # Initialise kernel
        self.initialise_mean_covar()
                                
        self.fit_GP(X, Y, rng_key)        
            
        
    def HMCModel(self, residuals, X, num_mixtures):
        """
        alpha: params for calculating lambda
        zetas: LOOCV variance for errors given the lambda values (in multinoial dist)
        """
        alphas = sample("alphas", dist.Normal(jnp.zeros((num_mixtures, self.n_input_params)),scale=jnp.ones((num_mixtures, self.n_input_params)) * jnp.sqrt(5)))
        zetas = sample("zetas", dist.TransformedDistribution(dist.LogNormal(jnp.ones(num_mixtures) * -1, jnp.ones(num_mixtures)), transforms.ComposeTransform([transforms.OrderedTransform()])))
        lambdas = self.jax_calc_lambdas(X, alphas, num_mixtures)
        s = numpyro.sample('s', dist.Categorical(lambdas)) 
        sample("residuals", dist.Normal(jnp.zeros(self.n_data), jnp.sqrt(zetas[s])), obs = residuals)
        
                
    
    def HMCGPModel(self, X, Y, alphas, zetas, num_mixtures):
        """
        taus: nugget values
        sigma2: variance of data
        deltas: lengthscale parameters for kernels
        betas: lengthscale parameters for mean
        """
        
        tau = self.nugget # deterministic("tau", jnp.asarray(self.nugget))
        sigma2 = sample("sigma2", dist.InverseGamma(jnp.ones(num_mixtures) * 2, 1))
        deltas = sample("deltas", dist.Gamma(4, jnp.ones((num_mixtures, self.n_input_params)) * 4))
        betas =  sample("betas", dist.Normal(loc=jnp.zeros(self.n_input_params + 1), scale=jnp.sqrt(10)))
    
        mean, kernel = self.calc_mean_kernel(X, alphas, deltas, betas, tau, sigma2, num_mixtures)
        sample("obs", dist.MultivariateNormal(loc=mean, covariance_matrix=kernel), obs=Y.T[0])
        
        
    def fit_mixtures(self, X, Y, residuals, rng_key):
        """Fits the Kernel Mixtures"""
        WAIC = torch.zeros(self.num_mixtures)
        alphas_list = [[] for _ in range(self.num_mixtures)]
        zetas_list = [[] for _ in range(self.num_mixtures)]
        
        for num_mix in range(1, self.num_mixtures + 1):
            HMC_kernel = MixedHMC(HMC(self.HMCModel))
            mcmc = MCMC(
                HMC_kernel,
                num_warmup=500,
                num_samples=500
            )
            mcmc.run(rng_key, residuals, X, num_mix)
            
            ## Get params fit gp + calc 
            posterior_samples = mcmc.get_samples()
            summary_dict = summary(posterior_samples, group_by_chain=False)
            alphas = summary_dict["alphas"]["mean"]
            zetas = summary_dict["zetas"]["mean"]
            
            # mcmc.print_summary()

            alphas_list[num_mix - 1] = jnp.asarray(alphas)
            zetas_list[num_mix - 1] = jnp.asarray(zetas)
            
            arv_mcmc = from_numpyro(mcmc)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                waic_summary = waic(arv_mcmc)
                WAIC[num_mix - 1] = -2 * (waic_summary.elpd_waic - waic_summary.p_waic)
        
        val = torch.argmax(-WAIC)
        model_mixtures = int(val + 1)
                        
        # Warn increase num mixtures
        if model_mixtures == self.num_mixtures:
            print("Increase number of mixtures")
        
        print(f"WAIC Scores: {WAIC}")
        print(f"The number of mixtures: {model_mixtures}")
                                            
        self.alphas = alphas_list[model_mixtures - 1]
        self.zetas = zetas_list[model_mixtures - 1]
        self.num_mixtures = model_mixtures
        
        
    def fit_GP(self, X, Y, rng_key):
        """ Fits the GP using the Kernel Mixtures """
        HMC_GPkernel = NUTS(self.HMCGPModel, max_tree_depth=5, dense_mass=False)
        mcmc = MCMC(
            HMC_GPkernel,
            num_warmup=100,
            num_samples=100
        )
        mcmc.run(rng_key, X, Y, self.alphas, self.zetas, self.num_mixtures)
        
        posterior_samples = mcmc.get_samples()
        summary_dict = summary(posterior_samples, group_by_chain=False)
        
        # mcmc.print_summary()
        
        self.betas = summary_dict["betas"]["mean"]
        self.deltas = summary_dict["deltas"]["mean"]
        self.noise = summary_dict["sigma2"]["mean"]
        self.mean = type(self.mean)(self.betas) 
        
    
    def calc_mean_kernel(self, X, alphas, deltas, betas, tau, sigma2, num_mixtures):
        """Calculates the Mean and Kernel for GP optimisation"""
        # Calculate Kernel
        X_nugget = self.jax_nugget_z(X, alphas, num_mixtures)
        X_lambdas = self.jax_calc_lambdas(X, alphas, num_mixtures)
        K = tau * X_nugget @ X_nugget.T * jnp.eye(self.n_data)# self.X_equal
        for i in range(num_mixtures):
            kern = type(self.kernel)(deltas[i])
            K += X_lambdas[:,i].reshape((X.shape[0], 1)) @ (X_lambdas[:,i]).reshape((1, X.shape[0])) * kern(X, X) + jnp.eye(X.shape[0]) * sigma2[i]
        
        # Calculate Mean
        mean_func = type(self.mean)(betas)
        mean = mean_func(X)
        return mean, K
    
    
    def jax_nugget_z(self, X, alphas, num_mixtures):
        """Calculate the mixture region assigned for the data"""
        z = jnp.zeros((X.shape[0], num_mixtures))
        for i, row in enumerate(self.jax_calc_lambdas(X, alphas, num_mixtures)):
            index = jnp.argmax(row)
            z = z.at[i, index].set(1)
        return z
    
    def jax_calc_lambdas(self, X, alphas, num_mixtures):
        """Calculate lambdas"""
        lambdas = []
        normalise = jnp.sum(jnp.exp((X @ alphas.T)), axis=1)
        for i, alpha in enumerate(alphas):
            lambdas.append((jnp.exp(X @ alpha.T).T / normalise))
            
        if num_mixtures == 1:
            return lambdas[0].reshape(-1,1)
        
        return jnp.stack(lambdas).T

    
    def LOO_residuals(self):
        """Calculates the LOO Resiguals of a stationary GP"""
        # Fit Initial GP
        mean = LinearMean
        kern = RBFKernel
        GP = GaussianProcess(mean, kern)
        GP.fit(self.X, self.Y) #, optimiser="LOOCV")
        
        # Get residuals
        loo_xpred = []
        loo_xsd = []
        for i in range(self.n_data):
            xpred, xvar = GP.LOOCV_loss(i)
            xsd = xvar.sqrt()
            loo_xpred += [xpred.item()]
            loo_xsd += [xsd.item()]
            
        residuals = (self.Y.T.detach().numpy()[0] - loo_xpred) / loo_xsd
                
        return residuals
    
    def calc_AIC(self, model, posterior, num_mixtures, residual, X):
        """Returns the AIC of the model"""
        ## ll returns dict, need to sum first key values
        ll = numpyro.infer.util.log_likelihood(
            model, 
            posterior,
            residual,
            X,
            num_mixtures
         )
        return num_mixtures * (1 + self.n_input_params) - 2 * ll
    
    def get_X_equal(self, X):
        """Caluclate X_equal (Usually X_equal is identity matrix)"""
        X_equal = jnp.broadcast_to(X, (X.shape[0], X.shape[0], self.n_input_params))
        XT_equal = jnp.moveaxis(X_equal, 0, 1)
        self.X_equal = jnp.equal(X_equal, XT_equal).all(axis=2) 
    
    def plot_1d(self, XX, plot_rows=1, plot_cols=1, plot_index=1):
        mean, covar = self.predict(XX)
        mean = mean.t().detach().numpy()[0]
        sd = torch.sqrt(torch.diagonal(covar)).detach().numpy()
        x = XX.t().detach().numpy()[0]
        X = self.destandardise_X(self.X, is_unit_length=False)
        if self.ynorm:
            Y = self.denormalise_Y(self.Y)
        else:
            Y = self.Y
            
        fig, ax = self.base_plot_1d(x, mean, sd, X, Y, plot_rows=plot_rows, plot_cols=plot_cols, plot_index=plot_index)
        return fig, ax
    
    def plot_2d(self, XX, plot_rows=1, plot_cols=1, plot_index=1):
        mean, _ = self.predict(XX)
        mean = mean.t().detach().numpy()[0]
        x = XX.detach().numpy()
        fig, ax = self.base_plot_2d(x, mean, plot_rows=plot_rows, plot_cols=plot_cols, plot_index=plot_index)
        return fig, ax

