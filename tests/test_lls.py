import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
import nsgpytorch

class ExactNSGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, inducing_points, likelihood):
        super(ExactNSGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        ls_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(nsgpytorch.kernels.GibbsRBFKernel(inducing_points, ls_kernel))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def common(train_x, train_y, test_x, name, training_iter, inducing_points, device):
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    inducing_points = inducing_points.to(device)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactNSGPModel(train_x, train_y, inducing_points, likelihood).to(device)

    # Find optimal model hyperparameters

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        # print(loss.item(), model.covar_module.base_kernel.inducing_ls[0].item(), 
        # model.covar_module.base_kernel.ls_likelihood.noise.item())
        optimizer.step()

    # Plot it
    f, ax = nsgpytorch.utils.plot_posterior(model, likelihood, train_x, train_y, 
                                            test_x, figsize=(16,6))

    f.savefig(os.path.join('tests/images', name + '.jpg'))

def test_sine_regression():
    training_iter = 100
    train_x = torch.linspace(-5,5,50).reshape(-1,1)
    train_y = torch.tensor(np.sin(train_x) + 0.2*np.random.normal(0, 1, size=train_x.shape)).ravel().to(train_x)
    test_x = torch.linspace(-5,5,100).reshape(-1,1).to(train_x)
    inducing_points = train_x[::4]
    device = "cuda"

    common(train_x, train_y, test_x, 'sine', training_iter, inducing_points, device)

def test_step_regression():
    import regdata as rd
    training_iter = 100
    train_x, train_y, test_x = rd.Step(backend='torch').get_data()
    inducing_points = train_x[::5]
    device = "cuda"

    common(train_x, train_y, test_x, 'step', training_iter, inducing_points, device)