# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:28:59 2024

@author: benmu
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import dgpsi as dgp

from BaseGP import BaseGP


class DeepGP(BaseGP):
    
    def __init__(self, layers=None, ynorm = True, optim_itter = 500):
        super().__init__(None, None, None, None, None, ynorm, None)
        self.layers = layers
        self.optim_itter = optim_itter
        self.model = None
        self.emulator = None
    
    def fit(self, X, Y):
        # Processing of data
        self.n_data, self.n_input_params = X.size()
        self.n_output_params = Y.size(1)
        XS, YS = self.initial_data_scale(X, Y)
        XS = XS.detach().numpy(); YS = YS.detach().numpy()
        
        if self.layers is None:
            self.layers = self.default_layers()
        
        # Fitting model
        self.model = dgp.dgp(XS, [YS], self.layers)
        self.model.train(N=self.optim_itter)
        
        # Saving emulator
        final_layer_obj = self.model.estimate()
        self.emulator = dgp.emulator(final_layer_obj)
        
    def predict(self, XX):
        # Scaling new inputs
        XXS = self.standardise_X(XX)
        XXS = XXS.detach().numpy()
        
        # predicting new values
        mean, var = self.emulator.predict(XXS, method='mean_var')
        mean = torch.from_numpy(mean); var = torch.from_numpy(var)
        
        # rescaling outputs
        if self.ynorm:
            mean = self.denormalise_Y(mean)
            var = var * torch.pow(self.Ystd, 2)
            
        return mean, var
    
    def default_layers(self):
        if self.n_input_params == 1:
            layer1 = [dgp.kernel(length=np.array([1. for _ in range(self.n_input_params)]), name='sexp')]
            layer2 = [dgp.kernel(length=np.array([1. for _ in range(self.n_input_params)]), name='sexp')]
            layer3 = [dgp.kernel(length=np.array([1. for _ in range(self.n_input_params)]), name='sexp', scale_est=True)]
            return dgp.combine(layer1, layer2, layer3)
        else:
            n = self.n_input_params
            layer1=[dgp.kernel(length=np.array([1]),name='sexp') for _ in range(n)]
            layer2=[dgp.kernel(length=np.array([1]),name='sexp',connect=np.arange(n)) for _ in range(n)]
            layer3=[dgp.kernel(length=np.array([1]),name='sexp',connect=np.arange(n)) for _ in range(n)]
            layer4=[dgp.kernel(length=np.array([1]),name='sexp',scale_est=True,connect=np.arange(n))]
            return dgp.combine(layer1,layer2,layer3,layer4)
            
    
    def plot_1d(self, XX, plot_rows=1, plot_cols=1, plot_index=1):
        mean, var = self.predict(XX)
        mean = mean.t().detach().numpy()[0]
        sd = torch.sqrt(var.t()[0]).detach().numpy()
        x = XX.t().detach().numpy()[0]
        X = self.destandardise_X(self.X)
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
        
        