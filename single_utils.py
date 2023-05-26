import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipydirect
import torch
import warnings
import gpytorch
import pandas as pd
from gpytorch.settings import fast_pred_var as gpt_settings
from gpytorch.constraints import Interval
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.settings import fast_pred_var
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from scipy.stats import norm
import botorch
from botorch.acquisition import (ExpectedImprovement, NoisyExpectedImprovement, UpperConfidenceBound,
                                  qExpectedImprovement)
from botorch.acquisition.objective import ScalarizedObjective
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.sampling import SobolQMCNormalSampler
from typing import Any, Iterator, List, Optional, Tuple, Union
from torch import Tensor
import scipy.io as sio
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from pyDOE2 import lhs
import openpyxl


class ExcelHandler:
    def __init__(self, filename):
        self.filename = filename
        self.variable_info = self._read_variable_info()
        self.feature_vars = self.variable_info[self.variable_info['Settings'] == 'Feature']
        self.constraint_vars = self.variable_info[self.variable_info['Settings'] == 'Constraint']
        self.nfeat = len(self.feature_vars)
        self.ncons = len(self.constraint_vars)

    def _read_variable_info(self):
        xl = pd.ExcelFile(self.filename)
        description_df = xl.parse('Description')
        return description_df

    def get_bounds(self):
        bounds_np = self.feature_vars[['Lower Bound', 'Upper Bound']].values
        bounds_tensor = torch.tensor(bounds_np, dtype=torch.float32)
        return bounds_tensor.t()

    def observations_to_tensors(self):
        # Read the 'Observations' sheet
        xl = pd.ExcelFile(self.filename)
        observations_df = xl.parse('Observations', header=0)

        # Extract input variables and responses
        features_df = observations_df[self.feature_vars['Variable Name'].values]
        responses_df = observations_df[self.variable_info['Variable Name'].values[len(self.feature_vars):]]

        # Separate responses into objective and constraints
        objective_df = responses_df.iloc[:, :1]
        constraints_df = responses_df.iloc[:, 1:]

        # Convert input variables and responses to PyTorch tensors
        train_x = torch.tensor(features_df.values, dtype=torch.float32)
        train_y = torch.tensor(objective_df.values, dtype=torch.float32).squeeze()
        cons = torch.tensor(constraints_df.values, dtype=torch.float32)

        return train_x, train_y, cons

    def load_excel(self):
        self.variable_info = self._read_variable_info()
        self.feature_vars = self.variable_info[self.variable_info['Settings'] == 'Feature']
        self.constraint_vars = self.variable_info[self.variable_info['Settings'] == 'Constraint']
        self.nfeat = len(self.feature_vars)
        self.ncons = len(self.constraint_vars)





class GPModel(GPyTorchModel, gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        if len(train_y.shape)==1:
            self._num_outputs = 1
        else:
            self._num_outputs = train_y.shape[-1]
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        #mean_x = mean_x.view(-1, 1)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BayesianOptimization:
    def __init__(self,train_X: Tensor,train_Y: Tensor,bounds: Tensor,noiseless_obs: bool,
        num_iter: int = 200,lr: float = 0.05):
        self.X = train_X
        self.Y = train_Y
        self.bounds = bounds
        self.noiseless_obs = noiseless_obs
        self.num_iter = num_iter
        self.lr = lr
        self.model, self.likelihood = self.train_GP()

    def get_posterior_stats(self, point):
        """Get the mean and standard deviation of the model's posterior at a given point."""
        self.model.eval()  # Set the model to evaluation mode
        posterior = self.model.posterior(point)
        mean = posterior.mean.detach()
        std_dev = posterior.variance.sqrt().detach()
        return mean, std_dev


    def get_training_posterior_stats(self):
        """Get the mean and standard deviation of the model's posterior at the training points."""
        means, std_devs = [], []
        for point in self.X:
            mean, std_dev = self.get_posterior_stats(point.view(1, -1))
            means.append(mean[0])
            std_devs.append(std_dev[0])

        means = torch.stack(means)
        std_devs = torch.stack(std_devs)
        means = means.squeeze()
        std_devs = std_devs.squeeze()

        return torch.stack((means, std_devs), dim=1)

    def train_GP(self,verbose=False):
        # Initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(lower_bound=1e-5, upper_bound=1e-4),
                )
        likelihood.raw_noise.requires_grad = True
        model = GPModel(self.X, self.Y, likelihood)
        # Train the model
        model.train()
        likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # Loss for GP - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        training_iterations = self.num_iter
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(self.X)
            loss = -mll(output, self.Y)
            loss.backward()
            if verbose:
                print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))
            optimizer.step()
        return model, likelihood

    def optimize_acquisition(self, acquisition_function=None, beta=None, num_restarts=5):
        dim = self.bounds.shape[-1]  # dimensionality of the problem
        best_f = self.Y.max().item()  # best observed value
        print(best_f)
        if acquisition_function is None:
            if beta is not None:
                # Use Upper Confidence Bound
                acq_func = UpperConfidenceBound(self.model, beta)
            else:
                # Use Expected Improvement
                acq_func = ExpectedImprovement(self.model, best_f=best_f)
         # Reshape bounds to 2 x d
        bounds = self.bounds.view(2, -1)
        candidate, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,  # number of candidates to generate (1 for single-point optimization)
            num_restarts=num_restarts,  # number of starting points for multistart optimization
            raw_samples=100,  # number of samples for initialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,  # use sequential optimization
        )

        return candidate, acq_value
