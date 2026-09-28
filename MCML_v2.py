# load data
import scipy.io as sio
from scipy import random, linalg
import torch
import numpy as np
from torch.autograd import Variable
UnDef = torch.tensor(0.0)


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


def compute_softmax_norm_i(W, inpt, i):
    softmax_norm = 0.
    N = inpt.size(1)
    for k in range(N):
        if i == k: continue
        exponent = W @ inpt[:, i] - W @ inpt[:, k]
        exponent = exponent.view(-1, 1)
        exponent = exponent.t() @ exponent
        softmax_norm += torch.exp(-exponent)
    return softmax_norm

def cal_pij(W, inpt, i, j):
    if i == j: return torch.tensor(0., device=device)  # since pij == 0
    zi = W @ inpt[:, i]
    zj = W @ inpt[:, j]
    exponent = zi - zj
    exponent = exponent.view(-1, 1)
    exponent = exponent.t() @ exponent
    pij = torch.exp(-exponent) / compute_softmax_norm_i(W, inpt, i)
    return pij

def cal_p0ij(labels, i, j):
    if i == j: return torch.tensor(0., device=device)
    if labels[i] == labels[j]:
        return torch.tensor(1., device=device)
    else: return torch.tensor(0., device=device)

def objective2(W, inpt, labels):
    loss = 0.
    N = inpt.size(1)
    for i in range(N):
        for j in range(N):
            if i ==j: continue
            if labels[i] != labels[j]: continue
            p0ij = cal_p0ij(labels, i, j)
            pij = cal_pij(W, inpt, i, j)
            l = p0ij * torch.log(pij)
            loss += l
    return - loss



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_train = sio.loadmat('mnist/mnist_train.mat')
data_test = sio.loadmat('mnist/mnist_test.mat')

X_train = torch.Tensor(data_train['train_X'].T).to(device)
rand_idx = torch.randperm(X_train.size(1))
X_train = X_train[:, rand_idx[0:10]]

labels_train = torch.Tensor(data_train['train_labels']).to(device)
labels_train = labels_train[rand_idx[0:10]]

X_test = torch.Tensor(data_test['test_X'].T).to(device)
rand_idx = torch.randperm(X_test.size(1))
X_test = X_test[:, rand_idx[0:10]]

labels_test = torch.Tensor(data_test['test_labels']).to(device)
labels_test = labels_test[rand_idx[0:10]]




# Initialization

W = Variable(0.01 * torch.rand(10, 784, device=device), requires_grad=True)


# Z_train = W @ X_train
# dA = cdist(Z_train.t(), Z_train.t())
# dA = cal_dA(X_train, A)
# print('random positive semi-define matrix for today is\n', A)
# P0 = cal_p0(X_train, labels_train)
# PA = cal_pA(dA)
# # Iterate
# epsilo = 0.001
#
#
#
# dA = cal_dA(X_train, A)
# PA = cal_pA(dA)
# gA = cal_gA(P0, PA, X_train)
epsilo = 0.001
optimizer = torch.optim.SGD([W], lr=0.001)
for i in range(10000):
    optimizer.zero_grad()

    # optimizer

    loss = objective2(W, X_train, labels_train)
    print('epoch:{}, W sum:{}, loss: {}'.format(i, torch.sum(W), loss))
    loss.backward(retain_graph=True)
    optimizer.step()

# A = A - epsilo * gA
# Lamda, U = torch.eig(A, eigenvectors=True)
# Lamda = Lamda[:,0]

