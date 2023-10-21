import torch


def log(X):
    S, U = torch.linalg.eigh(X)
    S = S.log().diag_embed()
    return U @ S @ U.transpose(-2, -1)


def exp(X):
    S, U = torch.linalg.eigh(X)
    S = S.exp().diag_embed()
    return U @ S @ U.transpose(-2, -1)


def sqrtm(X):
    S, U = torch.linalg.eigh(X)
    S = S.sqrt().diag_embed()
    return U @ S @ U.transpose(-2, -1)


def inv_sqrtm(X):
    S, U = torch.linalg.eigh(X)
    S = S.sqrt().reciprocal().diag_embed()
    return U @ S @ U.transpose(-2, -1)


def power(X, exponent):
    S, U = torch.linalg.eigh(X)
    S = S.pow(exponent).diag_embed()
    return U @ S @ U.transpose(-2, -1)


def geodesic(x, y, t):
    x_sqrt = torch.linalg.cholesky(x)
    x_sqrt_inv = x_sqrt.inverse()
    return (
        x_sqrt
        @ power(x_sqrt_inv @ y @ x_sqrt_inv.transpose(-2, -1), t)
        @ x_sqrt.transpose(-2, -1)
    )


def logm(x, y):
    c = torch.linalg.cholesky(x)
    c_inv = c.inverse()
    return c @ log(c_inv @ y @ c_inv.transpose(-2, -1)) @ c.transpose(-2, -1)


def expm(x, y):
    c = torch.linalg.cholesky(x)
    c_inv = c.inverse()
    return c @ exp(c_inv @ y @ c_inv.transpose(-2, -1)) @ c.transpose(-2, -1)
