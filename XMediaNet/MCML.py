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


def cal_p0(x, labels):
    Dl = cdist(labels, labels)
    I, J = np.where(Dl == 0)
    N = x.size(1)
    P0 = torch.zeros(N, N)
    for k in range(len(I)):
        P0[I[k], J[k]] = 1
    # set the diagonal element to NaN
    idxs = torch.arange(N).type(torch.IntTensor)
    for idx in idxs:
        P0[idx, idx] = UnDef
    return P0


def cal_pA(dA):
    N = dA.size(1)
    numerator = torch.exp(-dA)
    denominator = torch.sum(numerator) - torch.trace(numerator)
    PA = numerator / denominator
    # set the diagonal element to NaN
    idxs = torch.arange(N).type(torch.IntTensor)
    for idx in idxs:
        PA[idx, idx] = UnDef
    return PA

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
            p0ij = cal_p0ij(labels, i, j)
            pij = cal_pij(W, inpt, i, j)
            l = p0ij * torch.log(pij)
            loss += l
    return - loss

def cal_dA(x, A):
    N = x.size(1)
    dA = torch.zeros(N, N)

    for i in range(N):
        for j in range(N):
            tmp = (x[:, i] - x[:, j]).view(-1, 1)
            dA[i, j] = tmp.t() @ A @ tmp
    return dA


def cal_gA(P0, PA, x):
    r = x.size(0)
    gA = torch.zeros(r, r)
    N = x.size(1)
    for i in range(N):
        for j in range(N):
            temp = (x[:, j] - x[:, i]).view(-1, 1)
            # aug = (P0[i, j] - PA[i, j]) * temp @ temp.t()

            aug = (P0[i, j] - PA[i, j])
            gA = gA + aug
    return gA

def objective(x, labels):
    L1 = 0.0
    L2 = 0.0
    D = cdist(x.t(), x.t())
    N = x.size(1)
    for i in range(N):
        for j in range(N):
            if i != j and labels[i]==labels[j]:
                L1 = L1 + D[i,j]

    for i in range(N):
        Zi = 0
        for k in range(N):
            if i != k:
                Zi = Zi + torch.exp(-D[i, k])
        L2 = L2 + torch.log(Zi)
    loss = L1 + L2
    return loss





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

W = Variable(torch.rand(10, 784).to(device), requires_grad=True)
A = W.t() @ W

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

loss = objective2(W, X_train, labels_train)
loss.backward()
gA2 = A.grad

print ('ttttt')
# A = A - epsilo * gA
# Lamda, U = torch.eig(A, eigenvectors=True)
# Lamda = Lamda[:,0]

