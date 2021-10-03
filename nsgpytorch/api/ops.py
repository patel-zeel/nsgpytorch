import torch
import numpy as np
import gpytorch


def train(model, likelihood, mll, train_x, train_y, lr=0.1, iters=100,
          restarts=10, optimizer='adam', return_loss=True, seed=None, verbose=True):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = np.inf
    best_state_dict = None
    for restart in range(restarts):
        if return_loss:
            losses = []
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        for _ in range(iters):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            if return_loss:
                losses.append(loss.item())
        if verbose:
            print('Restart', restart, 'loss', loss.item())
        if loss < best_loss:
            best_state_dict = model.state_dict()
            best_loss = loss

    model.load_state_dict(best_state_dict)
    if return_loss:
        return losses


def test(model, likelihood, test_x, return_variance=True):
    model.eval()
    likelihood.eval()

    if return_variance:
        with gpytorch.settings.skip_posterior_variances(True):
            return likelihood(model(test_x)).mean
    else:
        with gpytorch.settings.fast_pred_var(True):
            pred_dist = likelihood(model(test_x))
            return pred_dist.mean, pred_dist.variance
