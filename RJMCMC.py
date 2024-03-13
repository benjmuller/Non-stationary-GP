# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:34:09 2023

@author: benmu
"""

# %% Imports

import numpy as np
import torch
import torch.distributions as dist

from scipy.stats import invgamma
import scipy
from math import prod

from Kernels import RBFKernel
from Tree import Tree
 

# %% RJMCMC Main

class TreedRJMCMC:
    
    def __init__(self, X, Y, n_itter, n_burnin, Xmin, Xmax):
        # Storing tree values
        self.trees = np.array([], dtype=Tree)
        
        # Data info
        self.X = X
        self.Y = Y
        self.n_data, self.n_input_params = X.size()
        self.rescale_Xmin = Xmin
        self.rescale_Xmax = Xmax
        self.Xmin = self.X.min(0).values.detach().numpy()
        self.Xmax = self.X.max(0).values.detach().numpy()
        self.split_locations, self.uniques = self.calc_possible_splits()
        self.conditions = None

        # parameters
        self.kernels = [] # list of kernel functions
        self.lambdas = [] # list of kernel lengthscale priors
        self.deltas = [] # kernel lengthscales
        self.g = [] # list of nugget params for kernels
        self.beta0 = None # beta0
        self.betas = [] # list linear mean coeficients
        self.W = None # list of Ws
        self.Winv = None # list of W's
        self.sigma2 = [] # list of sigmas
        self.tau2 = [] # list of taus
        
        # post fit values
        self.X_split = None # list of split X
        self.Y_split = None # list of split Y
        
        # Hyper Parameters
        self.min_num_xsplits = 1 # int(self.n_data / 20) # 0 is at least one x in split
        self.n_itter = n_itter
        self.n_burnin = n_burnin
        self.a = 0.5
        self.b = 2
        self.alphasig = 10
        self.qsig = 5
        self.alphatau = 10
        self.qtau = 5
        self.mu = np.array([0. for _ in range(self.n_input_params + 1)])
        self.B = np.eye(self.n_input_params + 1) 
        self.Binv = np.linalg.inv(self.B)
        self.rho = self.n_input_params + 1
        self.V = np.eye(self.n_input_params + 1)
        self.pd = np.array([1, 20, 10, 10]) # (1, 20, 10, 10)     

        
    def forward(self):   
        """Runs the RJMCMC"""
        self.new_GP(0, init=True) # adds the GP for the null / empty tree
        
        for i in range(self.n_itter + self.n_burnin):
            # Do a tree operation every 4 itterations
            if i < self.n_burnin:
                if (i + 1) % 4 == 0: self.tree_operations()
            # update GPs
            self.conditions = self.find_conditions(self.X)
                
            for j in range(len(self.kernels)):
                self.update_GP(j)
                
            self.global_update_gp()
            
            # if 10 * (i + 1) % (self.n_itter + self.n_burnin) == 0:
            #     print(f"{int(100 * (i + 1) / (self.n_itter + self.n_burnin))}% complete")
                
        # print("kernel", self.kernels, "\n")
        # print("deltas", self.deltas, "\n")
        # print("lambdas", self.lambdas, "\n")
        # print("g", self.g, "\n")
        # print("tau2", self.tau2, "\n")
        # print("sigma2", self.sigma2, "\n")
        # print("Winv", self.Winv, "\n")
        # print("W", self.W, "\n")
        # print("beta0", self.beta0, "\n")
        # print("beta", self.betas, "\n")
        # print("pd", self.pd)
        
        if self.trees.size > 0:
            print("Tree structure:")
            self.trees[0].print_rescaled(rescale_min=self.rescale_Xmin, rescale_max=self.rescale_Xmax)

    def tree_operations(self):
        """Suggest and perform tree operation"""
        prob = np.random.uniform()
        # if empty tree, only grow
        if self.trees.size == 0:
            self.tree_grow()
            return
             
        # if only root tree
        if self.trees.size == 1:
            if prob <= 0.25:
                self.tree_grow()
            elif prob > 0.25 and prob <= 0.5:
                self.tree_prune()
            else:
                self.tree_change()
            return
        
        # otherwise...
        if prob <= 0.2:
            self.tree_grow()
        elif prob > 0.2 and prob <= 0.4:
            self.tree_prune()
        elif prob > 0.4 and prob <= 0.8:
            self.tree_change()
        else:
            self.tree_rotate()
               
                
# %% Tree operations

    def tree_change(self):
        """
        1. Find valid tree changes
        2. sample (uniformly) the nodes and up or down movement
        3. calculate prob acceptance
        4. commit update to xval
        """
        trees = self.valid_tree_change()
        if len(trees) == 0: # no valid tree changes
            return
        prop_tree = trees[np.random.randint(0, len(trees))]
        prob = self.change_acceptance_ratio(prop_tree[0].depth)
        if prob > np.random.uniform():
            # Change accepted, change tree
            tree, loc = prop_tree
            unique_pos = np.where(self.uniques[tree.xdim] == tree.xval)[0][0]
            if loc == 0:
                # decrease in value
                tree.xval = self.uniques[tree.xdim][unique_pos - 1]
            else:
                # increase in value
                tree.xval = self.uniques[tree.xdim][unique_pos + 1]
        
    def calculate_marginal(self, n, X, Y, K, F, Kinv, deltas, lamb, g, tau2, sigma2, Winv, W, beta0, beta):
        Vbetainv = self.v_beta(F, Kinv, Winv, tau2)
        Vbeta = np.linalg.inv(Vbetainv)
        betahat = self.beta_hat(Vbeta, F, Kinv, Y, Winv, beta0, tau2)
        phi = self.calc_phi(Kinv, Y, beta0, betahat, Winv, Vbetainv, tau2)
        probK = self.joint_prob_dg(deltas, g, lamb)
        return self.marginal_Kv(K, Vbeta, n, W, tau2, self.n_input_params + 1, phi, probK)
        
    def tree_grow(self):
        """
        1. find leaf nodes
        2. sample (uniformly) from leaves
        3. calculate prob acceptance
        4. update tree
        """
        trees = self.valid_tree_grow()
        if self.trees.size > 0:
            # If there is a tree
            if len(trees) == 0: # No valid grows
                return
            prop_tree = trees[np.random.randint(0, len(trees))] # proposed grow
            
            # New tree
            tree, loc, possible_splits, possible_dims = prop_tree
            unique_dims = np.unique(possible_dims)
            xdim = unique_dims[np.random.randint(0, len(unique_dims))]
            possible_splits = np.array(possible_splits)
            possible_splits = possible_splits[np.where(possible_dims == xdim)]
            xval = possible_splits[np.random.randint(0, len(possible_splits))]
            
            ## Calculate Acceptance
            q = tree.depth
            proposed_GP = self.propose_GP()
            G = self.trees[0].num_empty() - 1
            P = self.trees[0].num_pruneable()
            if prop_tree[0].right is not None or prop_tree[0].left is not None:
                P += 1
                
            # Find GP and split locations
            if bool(loc):
                GP_index = tree.rightGP
            else:
                GP_index = tree.leftGP
                
            index = [condition[2] for condition in self.conditions].index(GP_index)
            X, indicies, _ = self.conditions[index]
            Y = self.Y[indicies,:].detach().numpy() 
            
        else:
            # If Null tree
            # New tree
            xdim = np.random.randint(0, self.n_input_params)
            xval = self.split_locations[xdim][np.random.randint(0, len(self.split_locations[xdim]))]
            
            # Calculate Acceptance
            q = 0
            proposed_GP = self.propose_GP()
            G = 1; P = 1
            X = self.X
            Y = self.Y.detach().numpy()
            indicies = np.arange(self.n_data)
            GP_index = 0
            
        # Calculate pK
        n = X.size(0)
        ntot = n
        F = self.calc_f(X).detach().numpy()
        K = self.kernels[GP_index](X, X) + torch.eye(n) * self.g[GP_index]
        Kinv = torch.linalg.inv(K).detach().numpy()
        K = K.detach().numpy()
        pK = self.calculate_marginal(n, X, Y, K, F, Kinv, self.deltas[GP_index], self.lambdas[GP_index], self.g[GP_index], self.tau2[GP_index], self.sigma2[GP_index], self.Winv, self.W, self.beta0, self.betas[GP_index])

        # Calculate pK1
        X1, X1_indicies, X2, X2_indicies = self.split_X(X, indicies, xdim, xval)
        n = X1.size(0)
        Y = self.Y[X1_indicies,:].detach().numpy()
        F = self.calc_f(X1).detach().numpy()
        K = self.kernels[GP_index](X1, X1) + torch.eye(n) * self.g[GP_index]
        Kinv = torch.linalg.inv(K).detach().numpy()
        K = K.detach().numpy()
        pK1 = self.calculate_marginal(n, X1, Y, K, F, Kinv, self.deltas[GP_index], self.lambdas[GP_index], self.g[GP_index], self.tau2[GP_index], self.sigma2[GP_index], self.Winv, self.W, self.beta0, self.betas[GP_index])
        
        # Calculate pK2
        n = X2.size(0)
        deltas, lamb, g, tau2, sigma2, beta = proposed_GP
        kernel = RBFKernel(deltas)
        Y = self.Y[X2_indicies,:].detach().numpy()
        F = self.calc_f(X2).detach().numpy()
        K = kernel(X2, X2) + torch.eye(n) * g
        Kinv = torch.linalg.inv(K).detach().numpy()
        K = K.detach().numpy()
        pK2 = self.calculate_marginal(n, X2, Y, K, F, Kinv, deltas, lamb, g, tau2, sigma2, self.Winv, self.W, self.beta0, beta)
        
        # Calculate qK2
        qK2 = self.joint_prob_dg(deltas, g, lamb)
        
        # Calculate Acceptance
        prob = self.grow_acceptance_ratio(G, P, q, pK1, pK2, pK, qK2) / ntot
            
        if prob > np.random.uniform():
            # Change accepted, change tree
            if self.trees.size > 0:
                # If tree is not null
                tree.grow(xdim, xval, loc)
                # Randomly assign GP
                probs = np.random.uniform()
                if probs > 0.5:
                    if bool(loc):
                        tree.right.leftGP = tree.rightGP
                        tree.rightGP = None
                        tree.right.rightGP = len(self.kernels)
                    else:
                        tree.left.leftGP = tree.leftGP
                        tree.leftGP = None
                        tree.left.rightGP = len(self.kernels)
                else:
                    if bool(loc):
                        tree.right.rightGP = tree.rightGP
                        tree.rightGP = None
                        tree.right.leftGP = len(self.kernels)
                    else:
                        tree.left.rightGP = tree.leftGP
                        tree.leftGP = None
                        tree.left.leftGP = len(self.kernels)
                    
                self.add_GP(proposed_GP, len(self.kernels))
                
                if bool(loc):
                    self.trees = np.insert(self.trees, len(self.trees), tree.right)
                else:
                    self.trees = np.insert(self.trees, len(self.trees), tree.left)
            else:
                # If null tree
                tree = Tree(xdim, xval)
                probs = np.random.uniform()
                if probs < 0.5:
                    tree.leftGP = 0
                    tree.rightGP = 1
                else:
                    tree.leftGP = 1
                    tree.rightGP = 0
                self.trees = np.insert(self.trees, 0, tree)
                self.add_GP(proposed_GP, 1)
        
        
    def tree_prune(self):
        """
        1. find leaf nodes
        2. sample (uniformly) from leaves
        3. calculate prob acceptance
        4. update tree
        """
        trees = self.valid_tree_prune()
        prop_tree = trees[np.random.randint(0, len(trees))]
        q = prop_tree.depth
        
        # Selecting GP to inherit from if prune accepted 
        probs = np.random.uniform()
        if probs > np.random.uniform():
            GP_index = prop_tree.rightGP
            other_GP_index = prop_tree.leftGP
        else:
            GP_index = prop_tree.leftGP
            other_GP_index = prop_tree.rightGP
        
        ## Calculate Acceptance
        G = self.trees[0].num_empty() - 1
        P = self.trees[0].num_pruneable()
            
        # Calculate pK        
        X, indicies = self.calc_partial_splits(self.X, prop_tree)
        n = X.size(0)
        ntot = n
        Y = self.Y[indicies,:].detach().numpy()
        F = self.calc_f(X).detach().numpy()
        K = self.kernels[GP_index](X, X) + torch.eye(n) * self.g[GP_index]
        Kinv = torch.linalg.inv(K).detach().numpy()
        K = K.detach().numpy()
        pK = self.calculate_marginal(n, X, Y, K, F, Kinv, self.deltas[GP_index], self.lambdas[GP_index], self.g[GP_index], self.tau2[GP_index], self.sigma2[GP_index], self.Winv, self.W, self.beta0, self.betas[GP_index])
        qK2 = self.joint_prob_dg(self.deltas[GP_index], self.g[GP_index], self.lambdas[GP_index])

        # Calculate pK1
        X1, X1_indicies, X2, X2_indicies = self.split_X(X, indicies, prop_tree.xdim, prop_tree.xval)
        n = X1.size(0)
        Y = self.Y[X1_indicies,:].detach().numpy()
        F = self.calc_f(X1).detach().numpy()
        K = self.kernels[GP_index](X1, X1) + torch.eye(n) * self.g[GP_index]
        Kinv = torch.linalg.inv(K).detach().numpy()
        K = K.detach().numpy()
        pK1 = self.calculate_marginal(n, X1, Y, K, F, Kinv, self.deltas[GP_index], self.lambdas[GP_index], self.g[GP_index], self.tau2[GP_index], self.sigma2[GP_index], self.Winv, self.W, self.beta0, self.betas[GP_index])
        
        # Calculate pK2
        n = X2.size(0)
        Y = self.Y[X2_indicies,:].detach().numpy()
        F = self.calc_f(X2).detach().numpy()
        K = self.kernels[other_GP_index](X2, X2) + torch.eye(n) * self.g[other_GP_index]
        Kinv = torch.linalg.inv(K).detach().numpy()
        K = K.detach().numpy()
        pK2 = self.calculate_marginal(n, X2, Y, K, F, Kinv, self.deltas[other_GP_index], self.lambdas[other_GP_index], self.g[other_GP_index], self.tau2[other_GP_index], self.sigma2[other_GP_index], self.Winv, self.W, self.beta0, self.betas[other_GP_index])
        
        # Calculate Acceptance
        prob = self.prune_acceptance_ratio(G, P, q, pK1, pK2, pK, qK2) * ntot
        
        if prob > np.random.uniform():
            # Change accepted, change tree
            # Saves storage location of proposed GP
            if prop_tree.parent is not None:
                if prop_tree.parent.left is prop_tree:
                    prop_tree.parent.leftGP = GP_index
                else:
                    prop_tree.parent.rightGP = GP_index
            
            # fixes GP storage locations   
            if prop_tree.leftGP == GP_index:
                prop_tree.minus_one_GPindex_greater_than(1)
            else:    
                prop_tree.minus_one_GPindex_greater_than(0)
            
            # Prunes tree
            if prop_tree.parent is not None:
                if prop_tree.parent.left is prop_tree:
                    prop_tree.parent.prune(0)
                else:
                    prop_tree.parent.prune(1)
        
            # Change tree
            self.trees = np.delete(self.trees, np.where(self.trees == prop_tree))
            self.delete_GP(other_GP_index)
            
    def tree_rotate(self):
        """
        1. find all possible rotations
        2. sample from possible rotations
        3. calculate prob acceptance
        4. update tree
        """
        trees = self.valid_tree_rotate()
        if len(trees) == 0: # No valid rotates
            return
        tree, swap_loc, loc = trees[np.random.randint(0, len(trees))]
        qDi = []; qDl = []
        qIi = []; qIl = []
        
        if bool(swap_loc):
            if bool(loc):
                if tree.right.left is not None:
                    qDi, qDl = tree.right.left.get_internals_and_leaves()
            else:
                if tree.right.right is not None:
                    qDi, qDl = tree.right.right.get_internals_and_leaves()
            if  tree.left is not None:
                qIi, qIl = tree.left.get_internals_and_leaves()
        else:
            if bool(loc):
                if tree.left.left is not None:
                    qDi, qDl = tree.left.left.get_internals_and_leaves()
            else:
                if tree.left.right is not None:
                    qDi, qDl = tree.left.right.get_internals_and_leaves()
            if tree.right is not None:
                qIi, qIl = tree.right.get_internals_and_leaves()
            
        
        qDi = np.array([tree.depth for tree in qDi])
        qDl = np.array([tree.depth for tree in qDl])
        qIi = np.array([tree.depth for tree in qIi])
        qIl = np.array([tree.depth for tree in qIl])
                
        prob = self.rotation_acceptance_ratio(qIi, qIl, qDi, qDl)
        if prob > np.random.uniform():
            # Change accepted, change tree
            if bool(swap_loc):
                if bool(loc):
                    tree.rightGP = tree.right.rightGP
                    tree.right.rightGP = None
                else:
                    tree.rightGP = tree.right.leftGP
                    tree.right.leftGP = None
            else:
                if bool(loc):
                    tree.leftGP = tree.left.rightGP
                    tree.left.rightGP = None
                else:
                    tree.leftGP = tree.left.leftGP
                    tree.left.leftGP = None
                    
            tree.rotate(swap_loc, loc)
        
        
# %% Initialise and update GPs
        
    def new_GP(self, index, init=False):
        if init:
            self.beta0 = self.init_sample_beta0()
            self.Winv = self.init_sample_Winv()
            self.W = np.linalg.inv(self.Winv)
        
        # Samples of parameters
        deltas = self.init_sample_delta()
        lamb = self.init_sample_lambda()
        g = self.init_sample_g(lamb)
        tau2 = self.init_sample_tau2()
        sigma2 = self.init_sample_sigma2()
        beta = self.init_sample_betas(self.beta0, sigma2, tau2, self.W)
        
        # Adding to stored values
        self.kernels.insert(index, RBFKernel(deltas))
        self.deltas.insert(index, deltas)
        self.lambdas.insert(index, lamb)
        self.g.insert(index, g)
        self.tau2.insert(index, tau2)
        self.sigma2.insert(index, sigma2)
        self.betas.insert(index, beta)
        
        
    def add_GP(self, items, index):
        deltas, lamb, g, tau2, sigma2, beta = items

        # Adding to stored values
        self.kernels.insert(index, RBFKernel(deltas))
        self.deltas.insert(index, deltas)
        self.lambdas.insert(index, lamb)
        self.g.insert(index, g)
        self.tau2.insert(index, tau2)
        self.sigma2.insert(index, sigma2)
        self.betas.insert(index, beta)
        
    def delete_GP(self, index):
        del self.kernels[index]
        del self.deltas[index]
        del self.lambdas[index]
        del self.g[index]
        del self.tau2[index]
        del self.sigma2[index]
        del self.betas[index]
         
    def propose_GP(self):
        # Samples of parameters
        deltas = self.init_sample_delta()
        lamb = self.init_sample_lambda()
        g = self.init_sample_g(lamb)
        tau2 = self.init_sample_tau2()
        sigma2 = self.init_sample_sigma2()
        beta = self.init_sample_betas(self.beta0, sigma2, tau2, self.W)
        return [deltas, lamb, g, tau2, sigma2, beta]
    
    def get_proposal_GP(self, index):
        deltas = self.deltas[index]
        lamb = self.lambdas[index]
        g = self.g[index]
        tau2 = self.tau2[index]
        sigma2 = self.sigma2[index]
        beta = self.betas[index]
        return [deltas, lamb, g, tau2, sigma2, beta]
    
    def global_update_gp(self):
        Vbeta0hat = self.v_beta0hat(self.Binv, self.Winv, self.sigma2, self.tau2)
        beta0hat = self.calc_beta0hat(Vbeta0hat, self.Binv, self.mu, self.Winv, self.betas, self.sigma2, self.tau2)
        Vwhat = self.V_what(self.betas, self.beta0, self.sigma2, self.tau2)
        
    
    def update_GP(self, i):  
        """MH Update GP"""
        # calculate MH samples
        gstar = self.sample_uniform(self.g[i])
        deltastar = self.sample_uniform(self.deltas[i])
      
        # calculate parameters for MH ratio
        if self.trees.size > 0:
            # If tree is not null
            index = [condition[2] for condition in self.conditions].index(i)
            X, indicies, _ = self.conditions[index]
            Y = self.Y[indicies,:].detach().numpy()
        else:
            # If null tree
            X = self.X
            Y = self.Y.detach().numpy()
            
        n = X.size(0)
        
        # calculate current
        K = self.kernels[i](X, X) + torch.eye(n) * self.g[i]
        F = self.calc_f(X).detach().numpy()
        Kinv = torch.linalg.inv(K).detach().numpy()
        K = K.detach().numpy()
        Vbeta0hat = self.v_beta0hat(self.Binv, self.Winv, self.sigma2, self.tau2)
        beta0hat = self.calc_beta0hat(Vbeta0hat, self.Binv, self.mu, self.Winv, self.betas, self.sigma2, self.tau2)
        bv = self.bv(self.betas[i], self.beta0, self.Winv, self.sigma2[i])
        Vbetainv = self.v_beta(F, Kinv, self.Winv, self.tau2[i]) 
        Vbeta = np.linalg.inv(Vbetainv)
        betahat = self.beta_hat(Vbeta, F, Kinv, Y, self.Winv, self.beta0, self.tau2[i])
        Vwhat = self.V_what(self.betas, self.beta0, self.sigma2, self.tau2)
        phi = self.calc_phi(Kinv, Y, self.beta0, betahat, self.Winv, Vbetainv, self.tau2[i])
        
        # Gibbs samples
        self.betas[i] = self.sample_beta(betahat, self.sigma2[i], Vbeta)
        self.tau2[i] = self.sample_tau2(self.n_input_params + 1, bv)
        self.sigma2[i] = self.sample_sigma2(n, phi)
        
        # update d        
        # MH samples
        Vbeta0hat = self.v_beta0hat(self.Binv, self.Winv, self.sigma2, self.tau2)
        beta0hat = self.calc_beta0hat(Vbeta0hat, self.Binv, self.mu, self.Winv, self.betas, self.sigma2, self.tau2)
        bv = self.bv(self.betas[i], self.beta0, self.Winv, self.sigma2[i])
        Vbetainv = self.v_beta(F, Kinv, self.Winv, self.tau2[i]) 
        Vbeta = np.linalg.inv(Vbetainv)
        betahat = self.beta_hat(Vbeta, F, Kinv, Y, self.Winv, self.beta0, self.tau2[i])
        phi = self.calc_phi(Kinv, Y, self.beta0, betahat, self.Winv, Vbetainv, self.tau2[i])
        pK = self.joint_prob_dg(self.deltas[i], self.g[i], self.lambdas[i])
        
        # Calculate Gibbs samples
        Kstar = RBFKernel(deltastar)(X, X) + torch.eye(n) * self.g[i]
        Kinvstar = torch.linalg.inv(Kstar).detach().numpy()
        Kstar = Kstar.detach().numpy()
        Vwhat = self.V_what(self.betas, self.beta0, self.sigma2, self.tau2)
        Winvstar = self.sample_Winv(Vwhat)
        Wstar = np.linalg.inv(Winvstar)
        Vbetainvstar = self.v_beta(F, Kinvstar, self.Winv, self.tau2[i]) 
        Vbetastar = np.linalg.inv(Vbetainvstar)
        betahatstar = self.beta_hat(Vbetastar, F, Kinvstar, Y, self.Winv, self.beta0, self.tau2[i])
        
        phistar = self.calc_phi(Kinvstar, Y, self.beta0, betahatstar, self.Winv, Vbetainvstar, self.tau2[i])
        pKstar = self.joint_prob_dg(deltastar, self.g[i], self.lambdas[i])

        # calculate mh / gibbs ratio
        mh1 = self.marginal_Kv(Kstar, Vbetastar, n, self.W, self.tau2[i], self.n_input_params + 1, phistar, pKstar)
        mh2 = self.marginal_Kv(K, Vbeta, n, self.W, self.tau2[i], self.n_input_params + 1, phi, pK)
        mhratio = np.exp(mh1 - mh2)
        prob = np.random.uniform()
        
        if prob < mhratio:
            # change accepted, update params
            self.kernels[i] = RBFKernel(deltastar)
            self.deltas[i] = deltastar
            mh = mh1
        else:
            mh = mh2
            
        ### Update g
        # Calculate Gibbs samples
        Kstar = self.kernels[i](X, X) + torch.eye(n) * gstar
        Kinvstar = torch.linalg.inv(Kstar).detach().numpy()
        Kstar = Kstar.detach().numpy()
        Vwhat = self.V_what(self.betas, self.beta0, self.sigma2, self.tau2)
        Vbetainvstar = self.v_beta(F, Kinvstar, self.Winv, self.tau2[i]) 
        Vbetastar = np.linalg.inv(Vbetainvstar)
        betahatstar = self.beta_hat(Vbetastar, F, Kinvstar, Y, self.Winv, self.beta0, self.tau2[i])
        phistar = self.calc_phi(Kinvstar, Y, self.beta0, betahatstar, self.Winv, Vbetainvstar, self.tau2[i])
        pKstar = self.joint_prob_dg(self.deltas[i], gstar, self.lambdas[i])
        
        # calculate mh / gibbs ratio
        mh1 = self.marginal_Kv(Kstar, Vbetastar, n, self.W, self.tau2[i], self.n_input_params + 1, phistar, pKstar)
        mhratio = np.exp(mh1 - mh)
        prob = np.random.uniform()
        
        if prob < mhratio:
            # change accepted, update params
            self.g[i] = gstar
            mh = mh1
            
        ## Update lambda
        lamb = self.sample_uniform(self.lambdas[i])
        
        # Calculate Gibbs samples
        pK = self.joint_prob_dg(self.deltas[i], self.g[i], self.lambdas[i])
        pKstar = self.joint_prob_dg(self.deltas[i], self.g[i], lamb)
    
        # calculate mh / gibbs ratio
        mhratio = np.exp(pKstar + dist.exponential.Exponential(1).log_prob(torch.from_numpy(np.array([lamb]))).detach().numpy() - pK - dist.exponential.Exponential(1).log_prob(torch.from_numpy(np.array([self.lambdas[i]]))).detach().numpy())
        prob = np.random.uniform()

        if prob < mhratio:
            # change accepted, update params
            self.lambdas[i] = lamb
            
        
# %% GP MH and Gibbs samples
        
    def init_sample_delta(self):
        return 0.5 * ((dist.gamma.Gamma(self.pd[0], self.pd[1]).sample((1,self.n_input_params)) + dist.gamma.Gamma(self.pd[2], self.pd[3]).sample((1,self.n_input_params))))[0].detach().numpy() 
        
    def init_sample_g(self, lamb):
        return np.random.exponential(lamb)
    
    def init_sample_beta0(self):
        return np.random.multivariate_normal(self.mu, self.B).reshape(-1,1)
    
    def init_sample_betas(self, beta0, sigma2, tau2, W):
        beta0 = beta0.T[0]
        betas = np.random.multivariate_normal(beta0, sigma2 * tau2 * W).reshape(-1,1)
        return betas
    
    def init_sample_tau2(self):
        return 1.
        
    def init_sample_sigma2(self):
        return invgamma(self.alphasig / 2, scale=self.qsig / 2).rvs()
    
    def init_sample_lambda(self):
        return 1.
    
    def init_sample_Winv(self):
        return np.eye(self.n_input_params + 1)
    
    def sample_beta(self, betavhat, sigma2, Vbetahat):
        betavhat = betavhat.T[0]
        return np.random.multivariate_normal(betavhat, sigma2 * Vbetahat, check_valid="ignore").reshape(-1,1)
    
    def sample_beta0(self, beta0hat, Vbeta0hat):
        beta0hat = beta0hat.T[0]
        return np.random.multivariate_normal(beta0hat, Vbeta0hat).reshape(-1,1)
        
    def sample_tau2(self, m, bv):
        return 1.
        
    def sample_Winv(self, Vwhat):
        return np.eye(self.n_input_params + 1)
        
    def sample_sigma2(self, n, phi):
        return float(1 / dist.gamma.Gamma(torch.Tensor([(self.alphasig + n) / 2]), torch.Tensor([(self.qsig + phi) / 2])).sample().detach().numpy())
    
    def sample_lambda(self):
        return np.random.exponential(1.)
        
        
    def sample_uniform(self, values):
        return np.random.uniform(3 * values / 4, 4 * values / 3)
    
    def sample_wishart(self, df, W):
        dim = W.shape[0]
        normal_samples = np.random.normal(0, 1, (dim, dim))
        sample_covariance = normal_samples @ normal_samples.T
        scaling_factor = df / np.random.chisquare(df)
        sample = scaling_factor * sample_covariance @ W
        return sample
    
# %% valid tree nodes for tree operations

    def valid_tree_change(self):
        """Calculates the valid tree change operations"""
        trees_plus = []
        trees_minus = []
        for tree in self.trees:
            valid_splits, indicies = self.calc_partial_splits(self.X, tree)
            if tree.xval != self.Xmax[tree.xdim]:
                upxval = self.uniques[tree.xdim][np.where(self.uniques[tree.xdim]==tree.xval)[0][0] + 1]
                if sum(valid_splits[:,tree.xdim] > upxval) > self.min_num_xsplits:
                    left_valid = True; right_valid = True
                    XL, indicies_L, XR, indicies_R = self.split_X(valid_splits, indicies, tree.xdim, upxval)
                    if tree.left is not None:
                        left_valid = self.are_splits_valid(XL, indicies_L, tree.left)
                    if tree.right is not None:
                        right_valid = self.are_splits_valid(XR, indicies_R, tree.right)
                    if left_valid and right_valid:
                        trees_plus.append((tree, 1))
            if tree.xval != self.Xmin[tree.xdim]:
                downxval = self.uniques[tree.xdim][np.where(self.uniques[tree.xdim]==tree.xval)[0][0] - 1]
                if sum(valid_splits[:,tree.xdim] < downxval) > self.min_num_xsplits:
                    left_valid = True; right_valid = True
                    XL, indicies_L, XR, indicies_R = self.split_X(valid_splits, indicies, tree.xdim, downxval)
                    if tree.left is not None:
                        left_valid = self.are_splits_valid(XL, indicies_L, tree.left)
                    if tree.right is not None:
                        right_valid = self.are_splits_valid(XR, indicies_R, tree.right)
                    if left_valid and right_valid:
                        trees_minus.append((tree, 0))
        return np.array(trees_plus + trees_minus)
    
    def valid_tree_grow(self):
        """Calculates the valid tree grow operations"""
        valid = []
        for tree in self.trees:
            if tree.left is None or tree.right is None:
                valid_splits = self.calc_partial_splits(self.X, tree)[0].detach().numpy()
            if tree.left is None:
                possible_splits = []
                possible_dims = []
                indicies = np.arange(valid_splits.shape[0])[np.where(valid_splits[:,tree.xdim] < tree.xval)]
                left_valid_splits = valid_splits[indicies]
                for i, dims in enumerate(self.split_locations):
                    for xsplit in dims:
                        if sum(left_valid_splits[:,i] < xsplit) > self.min_num_xsplits and sum(left_valid_splits[:,i] >= xsplit) > self.min_num_xsplits:
                            possible_splits.append(xsplit)
                            possible_dims.append(i)
                if len(possible_splits) > 0:
                    valid.append([tree, 0, possible_splits, possible_dims])
            if tree.right is None:
                possible_splits = []
                possible_dims = []
                indicies = np.arange(valid_splits.shape[0])[np.where(valid_splits[:,tree.xdim] >= tree.xval)]
                right_valid_splits = valid_splits[indicies]
                for i, dims in enumerate(self.split_locations):
                    for xsplit in dims:
                        if sum(right_valid_splits[:,i] < xsplit) > self.min_num_xsplits and sum(right_valid_splits[:,i] >= xsplit) > self.min_num_xsplits:
                            possible_splits.append(xsplit)
                            possible_dims.append(i)
                if len(possible_splits) > 0:
                    valid.append([tree, 1, possible_splits, possible_dims])
            
        return valid
    
    def valid_tree_prune(self):
        """Calculates the valid tree prune operations"""
        valid = []
        for tree in self.trees:
            if tree.right is None and tree.left is None:
                valid.append(tree)
        return np.array(valid)
        
    def valid_tree_rotate(self):
        """Calculates the valid tree rotate operations"""
        valid = []
        for tree in self.trees:
            if tree.left is not None:
                if tree.left.left is None:
                    if self.check_feasible_rotate(tree, 0, 0):
                        valid.append((tree, 0, 0))
                if tree.left.right is None:
                    if self.check_feasible_rotate(tree, 0, 1):
                        valid.append((tree, 0, 1))   
                    
            if tree.right is not None:
                if tree.right.left is None:
                    if self.check_feasible_rotate(tree, 1, 0):
                        valid.append((tree, 1, 0))
                if tree.right.right is None:
                    if self.check_feasible_rotate(tree, 1, 1):
                        valid.append((tree, 1, 1))
            
        return np.array(valid)
    
    def check_feasible_rotate(self, tree, swaploc, loc):
        if bool(swaploc):
            xdim, xval = tree.right.xdim, tree.right.xval
        else:
            xdim, xval = tree.left.xdim, tree.left.xval
            
        X, indicies = self.calc_partial_splits(self.X, tree)
        if bool(swaploc):
            X, indicies, _, _ = self.split_X(X, indicies, xdim, xval)
        else:
            _, _, X, indicies = self.split_X(X, indicies, xdim, xval)
        
        if X.size(0) == 0:
            return False
        
        return self.are_splits_valid(X, indicies, tree)
            

# %% Tree Operations acceptence ratios
        
    def change_acceptance_ratio(self, q):
        """Calculates acceptance ratio for tree change operation"""
        return self.a * (1 + q) ** -self.b
    

    def rotation_acceptance_ratio(self, qIi, qIl, qDi, qDl):
        """Calculates the acceptance ratio for tree rotation operation"""
        num1p1 = prod(self.a * (2 + qIi) ** -self.b)
        num1p2 = prod(1 - self.a * (2 + qIl) ** -self.b)
        den1p1 = prod(self.a * (1 + qIi) ** -self.b)
        den1p2 = prod(1 - self.a * (1 + qIl) ** -self.b)
        num2p1 = prod(self.a * qDi ** -self.b)
        num2p2 = prod(1 - self.a * qDl ** -self.b)
        den2p1 = prod(self.a * (1 + qDi) ** -self.b)
        den2p2 = prod(1 - self.a * (1 + qDl) ** -self.b)
        numerator = num1p1 * num1p2 * num2p1 * num2p2
        denominator = den1p1 * den1p2 * den2p1 * den2p2
        return numerator / denominator

    def grow_acceptance_ratio(self, G, P, q, pK1, pK2, pK, qK2):
        """Caluculates the tree acceptance ratios for grow and prune"""
        numerator = G * self.a * (1 + q) ** -self.b * (1 - self.a * (2 + q) ** -self.b) ** 2 
        denominator = P * (1 - self.a * (1 + q) ** -self.b)
        logprods = np.exp(pK1 + pK2 - pK - np.log(qK2))
        prob = logprods * numerator / denominator
        return prob
    
    def prune_acceptance_ratio(self, G, P, q, pK1, pK2, pK, qK2):
        """Caluculates the tree acceptance ratios for grow and prune"""
        numerator = G * self.a * (1 + q) ** -self.b * (1 - self.a * (2 + q) ** -self.b) ** 2
        denominator = P * (1 - self.a * (1 + q) ** -self.b)
        logprods = np.exp(pK + np.log(qK2) - pK1 - pK2)
        prob = logprods * numerator / denominator
        return prob
 
# %% GP Calculations

    def logdet(self, K):
        K = torch.from_numpy(K)
        chol = torch.linalg.cholesky(K)
        diag = torch.diag(chol).detach().numpy()
        return 2 * np.sum(np.log(diag))
        
    def marginal_Kv(self, K, Vbeta, n, W, tau2, m, phi, pK):
        """Marginal distribution for the kernel for mixture v""" 
        sign, logdetK = np.linalg.slogdet(K)
        _, logdetVbeta = np.linalg.slogdet(Vbeta)
                
        prod1 = 0.5 * (logdetVbeta - n * np.log(2 * np.pi) - logdetK - m * np.log(tau2) - np.log(np.linalg.det(W)))
        numprod2 = (self.alphasig / 2) * np.log((self.qsig / 2)) + (torch.lgamma(torch.tensor(0.5 * (self.alphasig + n)))).detach().numpy()
        denprod2 = (0.5 * (self.alphasig + n)) * np.log(0.5 * (self.qsig + phi)) + (torch.lgamma(torch.tensor(self.alphasig / 2))).detach().numpy()
        return np.log(pK) + prod1 + numprod2 - denprod2

    
    def calc_phi(self, Kinv, Z, beta0, beta, Winv, Vbetainv, tau2):
        phi = Z.T @ Kinv @ Z + beta0.T @ Winv @ beta0 / tau2 - beta.T @ Vbetainv @ beta
        return phi
    
    def joint_prob_dg(self, d, g, lamb):
        return (dist.exponential.Exponential(lamb).log_prob(torch.from_numpy(np.array([g]))).exp() * 0.5 * (torch.sum(dist.gamma.Gamma(self.pd[0], self.pd[1]).log_prob(torch.from_numpy(np.array(d))).exp() + dist.gamma.Gamma(self.pd[2], self.pd[3]).log_prob(torch.from_numpy(np.array(d))).exp()))).detach().numpy() 
               
    def v_beta(self, F, Kinv, Winv, tau2):
        return F.T @ Kinv @ F + Winv / tau2 
    
    def beta_hat(self, Vbeta, F, Kinv, Z, Winv, beta0, tau2):
        return (Vbeta @ (F.T @ Kinv @ Z + (Winv @ beta0 / tau2))).reshape(-1,1)
    
    def v_beta0hat(self, Binv, Winv, sigmas2, taus2):
        sums = 0
        for i in range(len(sigmas2)):
            sums += 1 / (sigmas2[i] * taus2[i])
        return np.linalg.inv(Binv + Winv * sums) 

    def calc_beta0hat(self, Vbeta0, Binv, mu, Winv, betas, sigmas2, taus2):
        vals = 0
        for i in range(len(sigmas2)):
            vals += betas[i] * (1 / (sigmas2[i] * taus2[i]))
        return Vbeta0 @ ((Binv @ mu).reshape(-1,1) + Winv @ vals)
    
    def bv(self, beta, beta0, Winv, sigma2): 
        return (beta - beta0).T @ Winv @ (beta - beta0) / sigma2
    
    def V_what(self, betas, betas0, sigmas2, taus2): 
        Vw = 0
        for i in range(len(betas)):
            Vw += (betas[i] - betas0) @ (betas[i] - betas0).T / (sigmas2[i] * taus2[i])
        return Vw
    
    def calc_R(self):
        if self.trees.size > 0:
            return self.trees[0].num_empty()
        else:
            return 1

# %% Finding Split locations

    def calc_possible_splits(self):
        """Calculate the possible split locations"""
        split_locations = []
        uniques = []
        for i in range(self.n_input_params):
            unique = torch.unique(self.X[:,i]).detach().numpy()
            split_locations.append(unique[1:(len(unique) - 1)])
            uniques.append(unique)
        return split_locations, uniques
    
    def find_conditions(self, XS):
        """Finds and splits the data based on the tree"""
        if self.trees.size == 0:
            return XS
        tree = self.trees[0]
        while tree.parent is not None:
            tree = tree.parent
        indices = torch.arange(0, XS.size(0))
        X_explore = [(XS, indices, tree)]
        X_list = []
        while X_explore: # While tree to explore
            temp_explore = []
            for X, indices_X, tree in X_explore:
                condition_L = X[:, tree.xdim] < tree.xval
                condition_R = X[:, tree.xdim] >= tree.xval
                XL = X[condition_L]; indices_L = indices_X[condition_L]
                XR = X[condition_R]; indices_R = indices_X[condition_R]
                                                               
                if tree.left is None:
                    X_list.append((XL, indices_L, tree.leftGP))
                else:
                    temp_explore.append((XL, indices_L, tree.left))
                
                if tree.right is None:
                    X_list.append((XR, indices_R, tree.rightGP))
                else:
                    temp_explore.append((XR, indices_R, tree.right))
                
            X_explore = temp_explore
        
        return X_list
    
    def are_splits_valid(self, XS, indicies, tree):
        """Finds if splits are valid"""
        X_explore = [(XS, indicies, tree)]
        X_list = []
        # is_broken = False
        while X_explore: # While tree to explore
            temp_explore = []
            for X, indicies_X, tree in X_explore:
                condition_L = X[:, tree.xdim] < tree.xval
                condition_R = X[:, tree.xdim] >= tree.xval
                if sum(condition_L) <= self.min_num_xsplits:
                    return False
                if sum(condition_R) <= self.min_num_xsplits:
                    return False
                indicies_L = indicies_X[condition_L]
                indicies_R = indicies_X[condition_R]
                XL = X[condition_L]
                XR = X[condition_R]
                                               
                if tree.left is None:
                    X_list.append((XL, indicies_L, tree.leftGP))
                else:
                    temp_explore.append((XL, indicies_L, tree.left))
                
                if tree.right is None:
                    X_list.append((XR, indicies_R, tree.rightGP))
                else:
                    temp_explore.append((XR, indicies_R, tree.right))
                
            X_explore = temp_explore
        
        return True
    
    def split_X(self, X, indicies_X, xdim, xval):
        condition_L = X[:, xdim] < xval
        condition_R = X[:, xdim] >= xval
        XL = X[condition_L]; indicies_L = indicies_X[condition_L]
        XR = X[condition_R]; indicies_R = indicies_X[condition_R]
        return XL, indicies_L, XR, indicies_R
    
    def calc_partial_splits(self, X, tree):
        indicies = np.arange(X.size(0))
        while tree.parent is not None:
            xdim, xval = tree.parent.xdim, tree.parent.xval
            if tree.parent.left is tree:
                X, indicies, _, _= self.split_X(X, indicies, xdim, xval)
            else:
                _, _, X, indicies = self.split_X(X, indicies, xdim, xval)
            tree = tree.parent
        return X, indicies
    
    def calc_f(self, X):
        """Calculate the X for the mean function"""
        ones = torch.ones((X.size(0), 1))
        return torch.cat([ones, X], dim = 1)   
    
    def print_gp(self, i):
        print("kernel", self.kernels[i], "\n")
        print("deltas", self.deltas[i], "\n")
        print("lambdas", self.lambdas[i], "\n")
        print("g", self.g[i], "\n")
        print("tau2", self.tau2[i], "\n")
        print("sigma2", self.sigma2[i], "\n")
        print("Winv", self.Winv[i], "\n")
        print("W", self.W[i], "\n")
        print("beta0", self.beta0[i], "\n")
        print("beta", self.betas[i], "\n")
        
    def normalise_XY(self, X, Y):
        Xmin = X.min(0).values
        Xmax = X.max(0).values
        XS = (X - Xmin) / (Xmax - Xmin)
        Ymean = Y.mean()
        Ystd = Y.std()
        YS = (Y - Ymean) / Ystd
        return XS, YS
    
    
    
# %% Predict

    def predict(self, XX):
        """Predicts the data XX"""
        predictions = np.zeros(XX.size(0))
        variances = np.zeros(XX.size(0))
        conditions_X = self.find_conditions(self.X)
        conditions = self.find_conditions(XX)
        if isinstance(conditions, torch.Tensor):
            X = self.X            
            F = self.calc_f(XX).detach().numpy()
            Fv = self.calc_f(X).detach().numpy()
            K = self.kernels[0](X, X) + torch.eye(self.n_data) * self.g[0]
            Kinv = torch.linalg.inv(K).detach().numpy()
            K = K.detach().numpy()
            Vbetainv = self.v_beta(Fv, Kinv, self.Winv, self.tau2[0])
            Vbeta = np.linalg.inv(Vbetainv)
            betahat = self.beta_hat(Vbeta, Fv, Kinv, self.Y.detach().numpy(), self.Winv, self.beta0, self.tau2[0])
            mean = F @ betahat + self.kernels[0](XX, X).detach().numpy() @ Kinv @ (self.Y.detach().numpy() - Fv @ betahat)
            Cinv = np.linalg.inv(K + self.tau2[0] * Fv @ self.W @ Fv.T) # + np.eye(self.n_data) * EPS)
            qv = self.kernels[0](X, XX).detach().numpy() + self.tau2[0] * Fv @ self.W @ F.T  
            kv = self.kernels[0](XX, XX).detach().numpy() + np.eye(XX.shape[0]) * self.g[0] + self.tau2[0] * F @ self.W @ F.T
            variances = np.diagonal(self.sigma2[0] * (kv - qv.T @ Cinv @ qv)).reshape(-1,1)
        else:
            for XX, indicies, i in conditions:
                pos_ind = [ind[2] for ind in conditions_X]
                loc = pos_ind.index(i)
                X = conditions_X[loc][0]
                indicies_X = conditions_X[loc][1]
                F = self.calc_f(XX).detach().numpy()
                Fv = self.calc_f(X).detach().numpy()
                K = self.kernels[i](X, X) + torch.eye(X.shape[0]) * self.g[i]
                Kinv = torch.linalg.inv(K).detach().numpy()
                K = K.detach().numpy()
                Vbetainv = self.v_beta(Fv, Kinv, self.Winv, self.tau2[i])
                Vbeta = np.linalg.inv(Vbetainv)
                betahat = self.beta_hat(Vbeta, Fv, Kinv, self.Y[indicies_X].detach().numpy(), self.Winv, self.beta0, self.tau2[i])
                mean = F @ betahat + self.kernels[i](XX, X).detach().numpy() @ Kinv @ (self.Y[indicies_X].detach().numpy() - Fv @ betahat)
                Cinv = np.linalg.inv(K + self.tau2[i] * Fv @ self.W @ Fv.T) # + np.eye(X.shape[0]) * EPS)
                qv = self.kernels[i](X, XX).detach().numpy() + self.tau2[i] * Fv @ self.W @ F.T          
                kv = self.kernels[i](XX, XX).detach().numpy() + np.eye(XX.shape[0]) * self.g[i] + self.tau2[i] * F @ self.W @ F.T
                var = np.diagonal(self.sigma2[i] * (kv - qv.T @ Cinv @ qv))
                predictions[indicies] = mean.T
                variances[indicies] = var.T 
            mean = predictions.reshape(-1,1)
            variances = variances.reshape(-1,1)
        return torch.from_numpy(mean), torch.from_numpy(variances)
             