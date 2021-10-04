import math
import torch
import gpytorch
from ..add_loss.gibbs_kernel_loss_term import GibbsKernelAddedLossTerm


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GibbsKernel(gpytorch.kernels.Kernel):
    r"""
    One Dimensional Gibbs Kernel.
    TODO: add equation and docstring here, think of generating autodocs as well.
    """

    def __init__(self,
                 inducing_points,
                 ls_kernel,
                 ard_num_dims: int = 1,
                 active_dims=None,
                 batch_shape=torch.Size([]), **kwargs):
        super().__init__(ard_num_dims=ard_num_dims,
                         active_dims=active_dims, batch_shape=batch_shape, **kwargs)

        self.inducing_points = inducing_points
        self.kwargs = kwargs

        self.register_parameter(name='raw_inducing_ls',
                                parameter=torch.nn.Parameter(torch.zeros_like(self.inducing_points).ravel()))
        self.register_constraint(
            'raw_inducing_ls', gpytorch.constraints.Positive())

        if self.kwargs['add_loss']:
            self.register_added_loss_term("inducing_gibbs_loss_term")

        self.ls_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.ls_model = ExactGPModel(
            self.inducing_points, torch.log(self.inducing_ls), self.ls_likelihood, ls_kernel)

    @property
    def inducing_ls(self):
        return self.raw_inducing_ls_constraint.transform(self.raw_inducing_ls)

    @inducing_ls.setter
    def inducing_ls(self, value):
        self.initialize(
            raw_inducing_ls=self.raw_inducing_ls_constraint.inverse_transform(value))

    def common_forward(self, x1, x2, diag=False, **params):
        r"""
        This part is common in all child classes/kernels.
        """

        if self.kwargs['add_loss'] and self.training:
            if not torch.equal(x1, x2):
                raise RuntimeError("x1 should equal x2 in training mode")
            new_added_loss_term = GibbsKernelAddedLossTerm(
                self.ls_likelihood,
                self.ls_model,
                self.inducing_points,
                # self.inducing_ls,
                # x1
            )
            self.update_added_loss_term(
                "inducing_gibbs_loss_term", new_added_loss_term)

        self.ls_model.eval()
        # Back propagation through posterior predictions:
        # https://github.com/cornellius-gp/gpytorch/issues/1691
        with gpytorch.settings.detach_test_caches(False), gpytorch.settings.skip_posterior_variances(True):
            self.ls_model.set_train_data(
                self.inducing_points, torch.log(self.inducing_ls), strict=False)
            lengthscale1 = self.get_ls(x1)
            lengthscale2 = self.get_ls(x2)

        return self._compute_prefix_and_Qij(x1, x2, lengthscale1, lengthscale2, diag)

    def get_ls(self, x):
        return torch.exp(self.ls_likelihood(self.ls_model(x)).mean)

    def _compute_prefix_and_Qij(self, x1, x2, l1, l2, diag):
        """
        Using Qij terminology from: https://proceedings.neurips.cc/paper/2003/file/326a8c055c0d04f5b06544665d8bb3ea-Paper.pdf
        """
        if self.active_dims is None:
            self.active_dims = torch.tensor(range(x1.shape[1]))
        x1_ = x1[:, self.active_dims]
        x2_ = x2[:, self.active_dims]
        l1_sq = l1.unsqueeze(-1).pow(2)
        l2_sq = l2.unsqueeze(-2).pow(2)
        l1l2_sq_sum = l1_sq + l2_sq
        sq_dist = self.covar_dist(
            x1_, x2_, diag=diag, square_dist=True, postprocess=False)

        prefix = (l1_sq.mul(l2_sq)).pow_(0.25).div_(
            l1l2_sq_sum.pow_(0.5)).mul_(math.sqrt(2))
        Qij = sq_dist.div_(l1l2_sq_sum)
        return prefix, Qij
