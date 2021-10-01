import gpytorch
from .gibbs_kernel import GibbsKernel

class GibbsRBFKernel(GibbsKernel):
    r"""
    One Dimensional Gibbs Kernel.
    TODO: add equation and docstring here, think of generating autodocs as well.
    """

    def forward(self, x1, x2, diag=False, **kwargs):
        prefix, Qij = self.common_forward(x1, x2, diag, **kwargs)

        return prefix.mul_((-Qij).exp_())