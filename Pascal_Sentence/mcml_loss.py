import torch

def cdist(x, y):
    '''
    x: n * dx
    y: m * dy
    '''

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)
    return dist


def comp_sim(Z_I, Z_T, device):
    Z = torch.cat((Z_I, Z_T), 0)
    N = Z.size(0)
    d = cdist(Z, Z)
    P = torch.exp(-d)
    mask = torch.ones_like(P, device=device) - torch.eye(N, device=device)
    P = P * mask
    eps = 1e-19 * N
    P[P < eps] = eps
    sumP = P.sum(1).unsqueeze(1).expand(N, N)
    P = P / sumP
    return P


def comp_logP(P, device):
    N = P.size(0)
    log_P = torch.log(P)
    mask = torch.ones_like(log_P, device=device) - torch.eye(N, device=device)
    log_P = log_P * mask
    return log_P


def comp_cost(Z_I, Z_T, labels, device):
    labels = torch.cat((labels, labels), 0)
    N = labels.size(0)
    P = comp_sim(Z_I, Z_T, device)
    logP = comp_logP(P, device)
    # find the indices that the labels are same
    dl = cdist(labels, labels)
    indices = (dl == 0).nonzero()

    mask = torch.zeros_like(logP, device=device)
    mask[indices[:, 0], indices[:, 1]] = 1.
    C = -logP * mask
    C = C.sum()

    # for i in range(L):
    #     C = C - logP[indices[i][0], indices[i][1]]
    C = C / N
    return C