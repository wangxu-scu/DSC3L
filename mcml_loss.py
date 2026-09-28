import torch
import numpy as np
import torch.nn.functional as F

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


def comp_sim(Z_I, Z_T, device, distribution='Normal'):
    Z = torch.cat((Z_I, Z_T), 0)
    N = Z.size(0)
    d = cdist(Z, Z)
    if distribution == 'Normal':
        delta = torch.tensor(1. / np.sqrt(2), device=device)
        delta.requires_grad = False
        P = torch.exp(-d / (delta ** 2 * 2))
    elif distribution == 'Student-t':
        v = 10.
        P = (1.0 + d/v)**(-(v+1)/2)
    elif distribution == 'Cauchy':
        gamma = 1.
        P = gamma / (d + gamma ** 2)
    mask = torch.ones_like(P, device=device) - torch.eye(N, device=device)
    P = P * mask
    eps = 1e-19 * N
    try:
        P[P < eps] = eps
    except:
        print(111)
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



def mcml_loss_compute(image_embeddings, text_embeddings, labels, f_divergence_type='KL'):
    Z = torch.cat((image_embeddings, text_embeddings), 0)
    N = Z.size(0)
    d = cdist(Z, Z)
    labels = torch.cat((labels, labels), 0)
    # label mask
    mylabels = labels.float()
    labelD = cdist(mylabels, mylabels)

    label_mask = (labelD < 0.5).float()  # 1-similar   0-dissimilar

    # softmax
    P = F.softmax(-d)
    # P = comp_sim(image_embeddings, text_embeddings, device = torch.device("cuda:0"), distribution='Normal')
    # normalize the true matching distribution
    label_mask = label_mask / torch.sum(label_mask, dim=1, keepdim=True)
    # KL Divergence
    p = (1e-8 + label_mask)
    q = (P + 1e-8)
    t = p / q
    if f_divergence_type == 'KL':
        f_t = t * torch.log(t)
        mcml_loss = torch.mean(torch.sum(f_t * q, 1))
        # mcml_loss = torch.mean(torch.sum(label_mask * torch.log(1e-8 + label_mask / (P + 1e-8)), 1))
    elif f_divergence_type == 'reverse_KL':
        f_t = -torch.log(t)
        mcml_loss = torch.mean(torch.sum(f_t * q, 1))
    elif f_divergence_type == 'X2_divergence':
        f_t = (t - 1) ** 2
        mcml_loss = torch.mean(torch.sum(f_t * q, 1))
    elif f_divergence_type == 'JS_divergence':
        m = (p + q) / 2.0
        t1 = p / m
        los_s_1 = torch.mean(torch.sum(t1 * torch.log(t1) * m, 1))
        t2 = q / m
        los_s_2 = torch.mean(torch.sum(t2 * torch.log(t2) * m, 1))
        mcml_loss = los_s_1 + los_s_2

    # mcml_loss = torch.mean(torch.sum(P * torch.log(1e-8 + P / (label_mask + 1e-8)), 1)) ## KL(P|Q)
    # mcml_loss = torch.mean(torch.sum(label_mask * torch.log((1e-8 + label_mask) / (P + 1e-8)), 1))  ## KL(Q|P)
    # mcml_loss = mcml_kl_loss
    return mcml_loss

# def mcml_loss_compute(image_embeddings, text_embeddings, labels):
#     Z = torch.cat((image_embeddings, text_embeddings), 0)
#     N = Z.size(0)
#     d = cdist(Z, Z)
#     labels = torch.cat((labels, labels), 0)
#     # label mask
#     mylabels = labels.float()
#     labelD = cdist(mylabels, mylabels)
#
#     label_mask = (labelD < 0.5).float()  # 1-similar   0-dissimilar
#
#     # softmax
#     P = F.softmax(-d)
#     # P = comp_sim(image_embeddings, text_embeddings, device = torch.device("cuda:0"), distribution='Normal')
#     # normalize the true matching distribution
#     label_mask = label_mask / torch.sum(label_mask, dim=1, keepdim=True)
#     # KL Divergence
#     p = (1e-9 + label_mask)
#     q = (P + 1e-9)
#     t = p / q
#     f_t = t * torch.log(t)
#     mcml_loss = torch.mean(torch.sum(f_t * q, 1))
#     return mcml_loss