from torchvision.datasets.folder import default_loader
import os
from torch.utils.data.dataset import Dataset
import torch
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA


class MyCustomDataset(Dataset):
    def __init__(self, dataset='wiki_shallow', state='train'):
        if 'few_labels' in dataset_name:
            dataset = dataset_name + '_K=' + str(gamma)
            data = sio.loadmat('../../NIPS/datasets/' + dataset)

            tr_img = data['tr_img'].astype('float32')
            tr_img = tr_img.reshape([tr_img.shape[0], -1])
            tr_txt = data['tr_txt'].astype('float32')
            tr_txt = tr_txt.reshape([tr_txt.shape[0], -1])
            tr_img_labels = data['tr_img_labels'].reshape([-1]).astype('int64')
            tr_txt_labels = data['tr_txt_labels'].reshape([-1]).astype('int64')

            tr_labeled_img = tr_img[tr_img_labels >= 0]
            tr_labeled_txt = tr_txt[tr_txt_labels >= 0]
            tr_labeled_img_labels = tr_img_labels[tr_img_labels >= 0]
            tr_labeled_txt_labels = tr_txt_labels[tr_txt_labels >= 0]

            tr_unlabeled_img = tr_img[tr_img_labels < 0]
            tr_unlabeled_txt = tr_txt[tr_txt_labels < 0]
            tr_unlabeled_img_labels = tr_img_labels[tr_img_labels < 0]
            tr_unlabeled_txt_labels = tr_txt_labels[tr_txt_labels < 0]

            I_train = np.concatenate([tr_labeled_img, tr_unlabeled_img], axis=0)
            T_train = np.concatenate([tr_labeled_txt, tr_unlabeled_txt], axis=0)
            labels_train = np.concatenate([tr_labeled_img_labels, tr_unlabeled_img_labels], axis=0)

            I_val = data['val_img'].astype('float32')
            I_val = I_val.reshape([I_val.shape[0], -1])
            T_val = data['val_txt'].astype('float32')
            T_val = T_val.reshape([T_val.shape[0], -1])
            # val_img_labels = data['val_img_labels'].reshape([-1]).astype('int64')
            labels_val = data['val_txt_labels'].reshape([-1]).astype('int64')
            I_test = data['te_img'].astype('float32')
            I_test = I_test.reshape([I_test.shape[0], -1])
            T_test = data['te_txt'].astype('float32')
            T_test = T_test.reshape([T_test.shape[0], -1])
            te_img_labels = data['te_img_labels'].reshape([-1]).astype('int64')
            labels_test = data['te_txt_labels'].reshape([-1]).astype('int64')


        elif dataset == 'wiki_shallow':
            data = sio.loadmat('wiki/wiki_feature.mat')
            self.I_tr = data['I_tr']
            self.T_tr = data['T_tr']
            self.I_tr_labels = data['I_tr_labels'].T
            self.I_te = data['I_te']
            self.T_te = data['T_te']
            self.I_te_labels = data['I_te_labels'].T
            self.I = np.concatenate((self.I_tr, self.I_te), 0)
            self.T = np.concatenate((self.T_tr, self.T_te), 0)
            self.labels = np.concatenate(
                (self.I_tr_labels, self.I_te_labels), 0)
            N = len(self.labels)
            # ind = np.random.permutation(N)
            classes = np.unique(self.labels)
            I_train, T_train, labels_train = [], [], []
            I_test, T_test, labels_test = [], [], []
            for c in classes:
                ind = (np.squeeze(self.labels, 1) == c).nonzero()
                I_train.append(self.I[ind][0:130])
                T_train.append(self.T[ind][0:130])
                labels_train.append(self.labels[ind][0:130])
                I_test.append(self.I[ind][130:])
                T_test.append(self.T[ind][130:])
                labels_test.append(self.labels[ind][130:])
            I_train = np.concatenate(I_train)
            T_train = np.concatenate(T_train)
            labels_train = np.concatenate(labels_train)
            I_test = np.concatenate(I_test)
            T_test = np.concatenate(T_test)
            labels_test = np.concatenate(labels_test)
        elif dataset == 'xmedianet':
            import h5py
            data = h5py.File('XMediaNet/xmedianet_deep_idx_data.h5py', 'r')
            I_train = data['train_imgs_deep'].value
            I_labels_train = data['train_imgs_labels'].value
            T_train = data['train_texts_idx'].value
            T_labels_train = data['train_texts_labels'].value
            assert (I_labels_train.any() == T_labels_train.any())
            I_test = data['test_imgs_deep'].value[0:4000]
            I_labels_test = data['test_imgs_labels'].value[0:4000]
            T_test = data['test_texts_idx'].value[0:4000]
            T_labels_test = data['test_texts_labels'].value[0:4000]
            assert (I_labels_test.any() == T_labels_test.any())
            labels_train = I_labels_train
            labels_test = I_labels_test
            self.wv_matrix = torch.from_numpy(data['wv_matrix'].value)

        elif dataset == 'nus_deep':
            import h5py
            data = h5py.File('/root/workspace/datasets/nus/nus_wide_deep_idx_data-corr-ae.h5py', 'r')
            I_train = data['train_imgs_deep'].value
            I_labels_train = data['train_imgs_labels'].value
            T_train = data['train_texts_idx'].value
            T_labels_train = data['train_texts_labels'].value
            assert (I_labels_train.any() == T_labels_train.any())
            I_test = data['test_imgs_deep'].value
            I_labels_test = data['test_imgs_labels'].value
            T_test = data['test_texts_idx'].value
            T_labels_test = data['test_texts_labels'].value

            # pca = PCA(n_components=512)
            # pca.fit(I_train)
            # I_train = pca.transform(I_train)
            # I_test = pca.transform(I_test)

            # I_valid = data['valid_imgs_deep'].value
            # I_labels_valid = data['valid_imgs_labels'].value
            # T_valid = data['valid_texts_idx'].value
            # T_labels_valid = data['valid_texts_labels'].value

            # I_test = np.concatenate((I_test, I_valid), 0)
            # T_test = np.concatenate((T_test, T_valid), 0)
            # I_labels_test = np.concatenate((I_labels_test, I_labels_valid), 0)
            # T_labels_test = np.concatenate((T_labels_test, T_labels_valid), 0)

            assert (I_labels_test.any() == T_labels_test.any())
            labels_train = I_labels_train
            labels_test = I_labels_test

            self.wv_matrix = data['wv_matrix'].value


        elif dataset == 'wiki_deep':
            import h5py
            data = h5py.File('../datasets/wiki_data/wiki_deep_idx_data-corr-ae.h5py', 'r')
            I_train = data['train_imgs_deep'].value
            I_labels_train = data['train_imgs_labels'].value
            T_train = data['train_texts_idx'].value
            T_labels_train = data['train_texts_labels'].value
            assert (I_labels_train.any() == T_labels_train.any())
            I_test = data['test_imgs_deep'].value
            I_labels_test = data['test_imgs_labels'].value
            T_test = data['test_texts_idx'].value
            T_labels_test = data['test_texts_labels'].value

            # pca = PCA(n_components=512)
            # pca.fit(I_train)
            # I_train = pca.transform(I_train)
            # I_test = pca.transform(I_test)

            # I_valid = data['valid_imgs_deep'].value
            # I_labels_valid = data['valid_imgs_labels'].value
            # T_valid = data['valid_texts_idx'].value
            # T_labels_valid = data['valid_texts_labels'].value

            # I_test = np.concatenate((I_test, I_valid), 0)
            # T_test = np.concatenate((T_test, T_valid), 0)
            # I_labels_test = np.concatenate((I_labels_test, I_labels_valid), 0)
            # T_labels_test = np.concatenate((T_labels_test, T_labels_valid), 0)

            assert (I_labels_test.any() == T_labels_test.any())
            labels_train = I_labels_train
            labels_test = I_labels_test

            self.wv_matrix = data['wv_matrix'].value

        elif dataset == 'wiki_deep_mcsm':

            train_img = sio.loadmat('../datasets/wiki_data_mcsm/data/i2t_attention_data/data/train_img.mat')
            I_train = train_img['train_img'] + 0.
            test_img = sio.loadmat('../datasets/wiki_data_mcsm/data/i2t_attention_data/data/test_img.mat')
            I_test = test_img['test_img'] + 0.
            train_lab = sio.loadmat('../datasets/wiki_data_mcsm/data/i2t_attention_data/data/train_lab.mat')
            I_labels_train = train_lab['train_lab'].astype(np.long)  - 1
            test_lab = sio.loadmat('../datasets/wiki_data_mcsm/data/i2t_attention_data/data/test_lab.mat')
            I_labels_test = test_lab['test_lab'].astype(np.long) - 1

            import h5py
            txt_data = h5py.File('../datasets/wiki_data_mcsm/data/i2t_attention_data/data/wiki_txt.hdf5', 'r')
            T_test = txt_data['test'].value - 1
            T_train = txt_data['train'].value - 1
            T_labels_test = txt_data['test_label'].value.astype(np.long) - 1
            T_labels_train = txt_data['train_label'].value.astype(np.long) - 1

            assert (I_labels_test.any() == T_labels_test.any())
            assert (I_labels_train.any() == T_labels_train.any())

            labels_train = T_labels_train
            labels_test = T_labels_test
            self.wv_matrix = txt_data['w2v'].value

        elif dataset == 'pascal_deep':
            import h5py
            data = h5py.File('../datasets/voc/pascal_deep_idx_data-corr-ae.h5py', 'r')
            I_train = data['train_imgs_deep'].value
            I_labels_train = data['train_imgs_labels'].value
            T_train = data['train_texts_idx'].value
            T_labels_train = data['train_texts_labels'].value
            assert (I_labels_train.any() == T_labels_train.any())
            I_test = data['test_imgs_deep'].value
            I_labels_test = data['test_imgs_labels'].value
            T_test = data['test_texts_idx'].value
            T_labels_test = data['test_texts_labels'].value
            assert (I_labels_test.any() == T_labels_test.any())
            labels_train = I_labels_train
            labels_test = I_labels_test

            self.wv_matrix = data['wv_matrix']

        if state == 'train':
            self.I = I_train
            self.T = T_train
            self.labels = labels_train

        if state == 'test':
            self.I = I_test
            self.T = T_test
            self.labels = labels_test

        self.I = torch.FloatTensor(self.I)

        if dataset == 'wiki_shallow':
            self.T = torch.FloatTensor(self.T)
        elif dataset == 'xmedianet' or dataset == 'nus_deep' or dataset == 'wiki_deep' or dataset == 'wiki_deep_mcsm':
            self.T = torch.LongTensor(self.T)

        self.labels = torch.LongTensor(self.labels)
        self.labels = self.labels.view(-1, 1)



    def __getitem__(self, index):
        I_item, T_item, label = self.I[index], self.T[index], self.labels[index]
        return I_item, T_item, label

    def __len__(self):
        count = len(self.I)
        assert len(
            self.I) == len(
            self.T) == len(
            self.labels)
        return count
