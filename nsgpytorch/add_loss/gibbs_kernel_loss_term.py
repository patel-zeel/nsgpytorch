#!/usr/bin/env python3

import torch
import math
from gpytorch.mlls.added_loss_term import AddedLossTerm
from gpytorch.utils.cholesky import psd_safe_cholesky


class GibbsKernelAddedLossTerm(AddedLossTerm):
    def __init__(self, likelihood, model, inducing_points):
        self.likelihood = likelihood
        self.model = model
        self.x = inducing_points

    def loss(self, *params):
        """
        Added loss
        """
        self.model.train()
        self.likelihood.train()

        covar = self.likelihood(self.model(self.x)).covariance_matrix
        cholesky = psd_safe_cholesky(covar)
        a = torch.sum(torch.log(cholesky.diagonal()))
        b = len(self.x)*torch.log(2*math.pi)
        return 0.5*(a+b)
