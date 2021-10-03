import torch
from sklearn.cluster import KMeans


def f_kmeans(tensor, n, seed=None):
    if seed:
        torch.manual_seed(seed)

    return torch.tensor(KMeans(n_clusters=n).fit(
        tensor.numpy()).cluster_centers_).to(tensor)


def f_random(tensor, n, seed=None):
    if seed:
        torch.manual_seed(seed)

    return tensor[tensor.multinomial(n)]
