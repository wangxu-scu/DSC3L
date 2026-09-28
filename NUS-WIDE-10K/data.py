import scipy.io as sio


def read_wiki():
    data = sio.loadmat('wiki/wiki_feature.mat')
    I_tr = data['I_tr']
    I_te = data['I_te']
    T_tr = data['T_tr']
    T_te = data['T_te']
    tr_labels = data['I_tr_labels']
    te_labels = data['I_te_labels']
    return I_tr, I_te, T_tr, T_te, tr_labels, te_labels
