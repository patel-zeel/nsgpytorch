import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
import nsgpytorch
import regdata as rd
rd.set_backend('torch')

# We will use the simplest form of GP model, exact inference


def common(train_x, train_y, test_x, name, inducing_points, device):
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    inducing_points = inducing_points.to(device)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    ls_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    kernel = gpytorch.kernels.ScaleKernel(
        nsgpytorch.kernels.GibbsRBFKernel(inducing_points, ls_kernel, **{'add_loss': True}))
    model = nsgpytorch.api.models.ExactGPModel(
        train_x, train_y, likelihood, kernel).to(device)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = nsgpytorch.api.ops.train(model, likelihood, mll, train_x, train_y, iters=2,
                                      restarts=2, optimizer='adam', return_loss=True, seed=0, verbose=True)

    # Plot it
    f, ax = nsgpytorch.utils.plotting.plot_posterior(model, likelihood, train_x, train_y,
                                                     test_x, figsize=(16, 6))

    ax[0].set_title('loss: '+str(losses[-1]))
    f.savefig(os.path.join('tests/images', name + '.jpg'))


def test_sinenoisy():
    file_name = 'sinenoisy'
    train_x, train_y, test_x = rd.SineNoisy().get_data()
    inducing_points = nsgpytorch.utils.inducing.f_kmeans(train_x, n=10, seed=0)
    device = "cpu"

    common(train_x, train_y, test_x, file_name, inducing_points, device)


def test_step():
    file_name = 'step'
    train_x, train_y, test_x = rd.Step().get_data()
    inducing_points = nsgpytorch.utils.inducing.f_kmeans(train_x, n=10, seed=0)
    device = "cpu"

    common(train_x, train_y, test_x, file_name, inducing_points, device)


def test_smooth1d():
    file_name = 'smooth1d'
    train_x, train_y, test_x = rd.Smooth1D().get_data()
    inducing_points = nsgpytorch.utils.inducing.f_kmeans(train_x, n=10, seed=0)
    device = "cpu"

    common(train_x, train_y, test_x, file_name, inducing_points, device)


def test_sinejump():
    file_name = 'sinejump'
    train_x, train_y, test_x = rd.SineJump1D().get_data()
    inducing_points = nsgpytorch.utils.inducing.f_kmeans(train_x, n=10, seed=0)
    device = "cpu"

    common(train_x, train_y, test_x, file_name, inducing_points, device)


def test_olympic():
    file_name = 'olympic'
    train_x, train_y, test_x = rd.Olympic().get_data()
    inducing_points = nsgpytorch.utils.inducing.f_kmeans(train_x, n=10, seed=0)
    device = "cpu"

    common(train_x, train_y, test_x, file_name, inducing_points, device)


def test_mcycle():
    file_name = 'mcycle'
    train_x, train_y, test_x = rd.MotorcycleHelmet().get_data()
    inducing_points = nsgpytorch.utils.inducing.f_kmeans(train_x, n=10, seed=0)
    device = "cpu"

    common(train_x, train_y, test_x, file_name, inducing_points, device)


def test_della_gatta():
    file_name = 'della_gatta'
    train_x, train_y, test_x = rd.DellaGattaGene().get_data()
    inducing_points = nsgpytorch.utils.inducing.f_kmeans(train_x, n=10, seed=0)
    device = "cpu"

    common(train_x, train_y, test_x, file_name, inducing_points, device)
