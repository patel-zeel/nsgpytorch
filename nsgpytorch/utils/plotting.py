import torch
import matplotlib.pyplot as plt
import gpytorch


def plot_posterior(model, likelihood, train_x, train_y, test_x, figsize):
    device = 'cpu'

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

        mean = observed_pred.mean.to(device)
        lower, upper = map(lambda x: x.to(device), observed_pred.confidence_region())
        inducing_points = model.covar_module.base_kernel.inducing_points.to(device)
        inducing_ls = model.covar_module.base_kernel.inducing_ls.to(device)
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        pred_ls = model.covar_module.base_kernel.get_ls(test_x).to(device)
        test_x = test_x.to(device)

        f, ax = plt.subplots(1, 2, figsize=figsize)
        # Plot training data as black stars
        ax[0].plot(train_x, train_y, 'k*')
        # Plot predictive means as blue line
        ax[0].plot(test_x, mean, 'b')
        # Shade between the lower and upper confidence bounds
        ax[0].fill_between(test_x.ravel(), lower, upper, alpha=0.5)
        ax[0].legend(['Observed Data', 'Mean', 'Confidence'])

        ax[1].plot(test_x, pred_ls)
        ax[1].scatter(inducing_points, inducing_ls)
        return f, ax