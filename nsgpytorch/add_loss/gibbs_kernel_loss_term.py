#!/usr/bin/env python3

import torch
import math
from gpytorch.mlls.added_loss_term import AddedLossTerm
from gpytorch.utils.cholesky import psd_safe_cholesky
torch.pi = torch.tensor(math.pi)


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
        b = len(self.x)*torch.log(2*torch.pi)
        return 0.5*(a+b)

        # Sahil's version
        # # Setting posterior mode
        # self.ls_model.eval()
        # self.ls_likelihood.eval()

        # with gpytorch.settings.detach_test_caches(False):
        #     self.ls_model.set_train_data(
        #         self.inducing_points, torch.log(self.inducing_ls), strict=False)
        #     output = self.ls_likelihood(
        #         self.ls_model(self.x))  # predictive posterior
        #     ls_pred_mean = torch.exp(output.mean)
        #     loss = -output.log_prob(ls_pred_mean)

        # return loss
