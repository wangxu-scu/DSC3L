# load data
import scipy.io as sio
from scipy import random, linalg
import torch
import numpy as np
from torch.autograd import Variable
UnDef = torch.tensor(0.0)
torch.manual_seed(2)

def np_lda(train, labels):
    if type(train) is list:
        data = np.concatenate(train, axis=0)
        la = np.concatenate(labels, axis=0)
    else:
        data = train
        la = labels

    classes = np.unique(la)
    m, n = data.shape
    cnum = classes.shape[0]
    m_all = np.reshape(np.mean(data, axis=0), [1, -1])
    sw = np.zeros([n, n])
    sb = np.zeros([n, n])
    for i in range(cnum):
        inx = np.where(np.equal(la, classes[i]))
        x_i = np.reshape(data[inx, :], [-1, n])
        m_i = np.reshape(np.mean(x_i, axis=0), [1, -1])
        tmp1 = x_i - m_i
        tmp2 = m_i - m_all
        sw += np.dot(tmp1.T, tmp1)
        sb += x_i.shape[0] * np.dot(tmp2.T, tmp2)

    eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(sw + np.eye(n, n) * 1e-3), sb))
    inx = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inx]
    eigval = eigval[inx]
    return eigvec[:, 0: 9], eigval



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

def comp_sim(W, X):
    N = X.size(0)
    # compute projected data
    Z = X @ W
    d = cdist(Z, Z)
    P = torch.exp(-d)
    mask = torch.ones_like(P, device=device) - torch.eye(N, device=device)
    P = P * mask
    eps = 1e-19 * N
    P[P<eps] = eps
    sumP = P.sum(1).unsqueeze(1).expand(N,N)
    P = P / sumP
    return P

def comp_logP(P):
    N = P.size(0)
    log_P = torch.log(P)
    mask = torch.ones_like(log_P, device=device) - torch.eye(N, device=device)
    log_P = log_P * mask
    return log_P

def comp_cost(P, labels):
    N = labels.size(0)
    C = 0
    logP = comp_logP(P)
    # find the indices that the labels are same
    dl = cdist(labels, labels)
    indices = (dl==0).nonzero()
    L = indices.size(0)

    mask = torch.zeros_like(logP, device=device)
    mask[indices[:,0], indices[:,1]] = 1.
    C = -logP * mask
    C = C.sum()

    # for i in range(L):
    #     C = C - logP[indices[i][0], indices[i][1]]
    C = C / N
    return C

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# data_train = sio.loadmat('mnist/mnist_train.mat')
# data_test = sio.loadmat('mnist/mnist_test.mat')
#
# X_train = data_train['train_X']
# rand_idx = np.random.permutation(X_train.shape[0])
# X_train = X_train[rand_idx[0:1000],:]
#
# labels_train = data_train['train_labels']
# labels_train = labels_train[rand_idx[0:1000]]
#
# X_test = data_test['test_X']
# rand_idx = np.random.permutation(X_test.shape[0])
# X_test = X_test[rand_idx[0:1000],:]
#
# labels_test = data_test['test_labels']
# labels_test = labels_test[rand_idx[0:1000]]


data = sio.loadmat('wiki/wiki_feature.mat')
X_train = data['I_tr']
# X_train = data['T_tr']
labels_train = data['I_tr_labels'].T

X_test= data['I_te']
# X_test = data['T_te']
labels_test = data['I_te_labels'].T

### classificaition before MCML
# from sklearn.decomposition import PCA
# pca = PCA(n_components=50)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)





from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, labels_train)
acc = (neigh.predict(X_test)==labels_test.squeeze(1)).sum() / 1000.
print ('acc before MCML:{}'.format(acc))



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(n_components=9)
clf.fit(X_train, labels_train)
L_train = clf.transform(X_train)
L_test = clf.transform(X_test)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(L_train, labels_train)
acc = (neigh.predict(L_test)==labels_test.squeeze(1)).sum() / 1000.
print ('acc of LDA:{}'.format(acc))


X_train = torch.Tensor(X_train).to(device)
X_test = torch.Tensor(X_test).to(device)
labels_train = torch.Tensor(labels_train).to(device)
labels_test = torch.Tensor(labels_test).to(device)

# Initialization
W = 0.01 * torch.rand(X_train.size(1), 9, device=device)
W.requires_grad = True

# optimizer
# optimizer = torch.optim.LBFGS([W])
# for i in range(100):
#     def closure():
#         optimizer.zero_grad()
#         P = comp_sim(W, X_train)
#         loss = comp_cost(P, labels_train)
#         loss.backward()
#         return loss
#     loss = closure()
#     print('epoch:{}, W sum:{}, loss: {}'.format(i, torch.sum(W), loss))
#     optimizer.step(closure)

optimizer = torch.optim.Adam([W],lr=0.001)
for i in range(1000):
    optimizer.zero_grad()
    P = comp_sim(W, X_train)
    loss = comp_cost(P, labels_train)
    print('epoch:{}, W sum:{}, loss: {}'.format(i, torch.sum(W), loss))
    loss.backward(retain_graph=True)
    optimizer.step()

### classificaition after MCML
Z_train = X_train @ W
Z_test = X_test @ W
Z_train = Z_train.cpu().detach().numpy()
Z_test = Z_test.cpu().detach().numpy()
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(Z_train, labels_train)
acc = (neigh.predict(Z_test)==labels_test.squeeze(1)).sum().numpy() / 1000.
print ('acc after MCML:{}'.format(acc))

