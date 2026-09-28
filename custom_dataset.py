from torchvision.datasets.folder import default_loader
import os
from torch.utils.data.dataset import Dataset
import torch
import scipy.io as sio
import h5py
import numpy as np
from sklearn.decomposition import PCA


class MyCustomDataset(Dataset):
    def __init__(self, dataset='wiki_shallow', state='train'):
        if dataset == 'wiki_shallow':
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
            self.wv_matrix = data['wv_matrix'].value

        elif dataset == 'nus':
            import h5py
            data = h5py.File(
                '../datasets/nus/nus_wide_deep_idx_data-corr-ae.h5py', 'r')
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

        elif dataset == 'wiki_deep_corr-ae':
            import h5py
            data = h5py.File(
                '../datasets/wiki_data/wiki_deep_idx_data-corr-ae.h5py', 'r')
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

            self.wv_matrix = data['wv_matrix'].value

        elif dataset == 'wiki_deep':
            train_img = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/train_img.mat')
            I_train = train_img['train_img']
            train_txt = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/train_txt.mat')
            T_train = train_txt['train_txt']
            I_labels_train = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/train_img_lab.mat')
            I_labels_train = I_labels_train['train_lab'].astype(np.long) - 1
            T_labels_train = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/train_txt_lab.mat')
            T_labels_train = T_labels_train['train_lab'].astype(np.long) - 1

            test_img = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/test_img.mat')
            test_txt = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/test_txt.mat')
            test_img_lab = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/test_img_lab.mat')
            test_txt_lab = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/test_txt_lab.mat')

            I_test = test_img['test_img']
            test_size = I_test.shape[0]

            # indice = sio.loadmat(
            #     '/root/workspace/datasets/wiki_data/extracted_features/indice.mat')
            # indice = indice['indice'].reshape(-1,)
            #
            # indice = np.load('/root/workspace/datasets/wiki_data/extracted_features/indices_corr_ae.npy')
            # test_idx = indice[231:]
            # tune_idx = indice[:231]

            # idx = np.random.permutation(np.arange(test_size))

            idx = np.load(
                '/root/workspace/datasets/wiki_data/extracted_features/my_indices.npy')
            test_idx = idx[:462]
            tune_idx = idx[462:]

            # indice = sio.loadmat( '/root/workspace/datasets/wiki_data/extracted_features/indice.mat')
            # idx = indice['indice'].reshape(-1,)
            # tune_idx = idx[0:231]
            # test_idx = idx[231:]

            I_test = test_img['test_img'][test_idx]
            I_tune = test_img['test_img'][tune_idx]
            labels_test = test_img_lab['test_lab'].astype(np.long) - 1
            I_labels_test = labels_test[test_idx]
            I_labels_tune = labels_test[tune_idx]

            T_test = test_txt['test_txt'][test_idx]
            T_tune = test_txt['test_txt'][tune_idx]
            labels_test = test_txt_lab['test_lab'].astype(np.long) - 1
            T_labels_test = labels_test[test_idx]
            T_labels_tune = labels_test[tune_idx]

            assert (I_labels_test.any() == T_labels_test.any())
            assert (I_labels_train.any() == T_labels_train.any())

            labels_train = I_labels_train
            labels_test = I_labels_test

            from sklearn import preprocessing
            img_scaler = preprocessing.StandardScaler().fit(I_train)
            I_train = img_scaler.transform(I_train)
            I_test = img_scaler.transform(I_test)

            txt_scaler = preprocessing.StandardScaler().fit(T_train)
            T_train = txt_scaler.transform(T_train)
            T_test = txt_scaler.transform(T_test)

            # sio.savemat('/root/workspace/datasets/wiki_data/extracted_features/wiki_features.mat',
            #             {'X1': I_train,
            #              'X2': T_train,
            #              'XTe1': I_test,
            #              'XTe2': T_test,
            #              'XV1': I_tune,
            #              'XV2': T_tune,
            #              'trainLabel': T_labels_train,
            #              'testLabel': T_labels_test,
            #              'tuneLabel': T_labels_tune})
            # self.wv_matrix = txt_data['w2v'].value

        elif dataset == 'my_wiki_deep':
            train_img = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/my_extracted/train_img.mat')
            I_train = train_img['train_img']
            train_txt = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/my_extracted/train_txt.mat')
            T_train = train_txt['train_txt']
            I_labels_train = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/my_extracted/train_img_lab.mat')
            I_labels_train = I_labels_train['train_img_lab'].reshape(
                -1, 1).astype(np.long)
            T_labels_train = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/my_extracted/train_txt_lab.mat')
            T_labels_train = T_labels_train['train_lab'].astype(np.long) - 1

            test_img = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/my_extracted/test_img.mat')
            test_txt = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/my_extracted/test_txt.mat')
            test_img_lab = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/my_extracted/test_img_lab.mat')
            test_txt_lab = sio.loadmat(
                '/root/workspace/datasets/wiki_data/extracted_features/my_extracted/test_txt_lab.mat')

            # indice = sio.loadmat(
            #     '/root/workspace/datasets/wiki_data/extracted_features/indice.mat')
            # indice = indice['indice'].reshape(-1,)
            #
            # indice = np.load('/root/workspace/datasets/wiki_data/extracted_features/indices_corr_ae.npy')
            # test_idx = indice[231:]
            # tune_idx = indice[:231]

            # idx = np.random.permutation(np.arange(test_size))

            idx = np.load(
                '/root/workspace/datasets/wiki_data/extracted_features/my_indices.npy')
            test_idx = idx[:462]
            tune_idx = idx[462:]

            # indice = sio.loadmat( '/root/workspace/datasets/wiki_data/extracted_features/indice.mat')
            # idx = indice['indice'].reshape(-1,)
            # tune_idx = idx[0:231]
            # test_idx = idx[231:]

            I_test = test_img['test_img'][test_idx]
            I_tune = test_img['test_img'][tune_idx]
            labels_test = test_img_lab['test_img_lab'].reshape(
                -1, 1).astype(np.long)
            I_labels_test = labels_test[test_idx]
            I_labels_tune = labels_test[tune_idx]

            T_test = test_txt['test_txt'][test_idx]
            T_tune = test_txt['test_txt'][tune_idx]
            labels_test = test_txt_lab['test_lab'].astype(np.long) - 1
            T_labels_test = labels_test[test_idx]
            T_labels_tune = labels_test[tune_idx]

            assert (I_labels_test.any() == T_labels_test.any())
            assert (I_labels_train.any() == T_labels_train.any())

            labels_train = I_labels_train
            labels_test = I_labels_test
            print()

        elif dataset == 'pascal_deep':
            data_dir = '/root/workspace/datasets/pascal/my_extracted_feature/'
            train_img = sio.loadmat(data_dir + 'train_img.mat')
            I_train = train_img['train_img']
            train_txt = sio.loadmat(data_dir + 'train_txt.mat')
            T_train = train_txt['train_txt']

            test_img = sio.loadmat(data_dir + 'test_img.mat')
            I_test = test_img['test_img']
            test_txt = sio.loadmat(data_dir + 'test_txt.mat')
            T_test = test_txt['test_txt']

            train_img_lab = sio.loadmat(data_dir + 'train_img_lab.mat')
            labels_train = train_img_lab['train_img_lab']
            test_img_lab = sio.loadmat(data_dir + 'test_img_lab.mat')
            labels_test = test_img_lab['test_img_lab']

            # from sklearn import preprocessing
            # img_scaler = preprocessing.StandardScaler().fit(I_train)
            # I_train = img_scaler.transform(I_train)
            # I_test = img_scaler.transform(I_test)
            #
            # txt_scaler = preprocessing.StandardScaler().fit(T_train)
            # T_train = txt_scaler.transform(T_train)
            # T_test = txt_scaler.transform(T_test)

            # sio.savemat('/root/workspace/datasets/pascal/my_extracted_feature/pascal_features.mat',
            #             {'X1': I_train,
            #              'X2': T_train,
            #              'XTe1': I_test,
            #              'XTe2': T_test,
            #              'trainLabel': labels_train,
            #              'testLabel': labels_test})

        elif dataset == 'xmedianet_deep':
            data_dir = '/root/workspace/dataset/XmediaNet/my_extracted_feature/'
            train_img = sio.loadmat(data_dir + 'train_img.mat')
            I_train = train_img['train_img']
            train_txt = sio.loadmat(data_dir + 'train_txt.mat')
            T_train = train_txt['train_txt']

            test_img = sio.loadmat(data_dir + 'test_img.mat')
            I_test = test_img['test_img']
            test_txt = sio.loadmat(data_dir + 'test_txt.mat')
            T_test = test_txt['test_txt']

            train_img_lab = sio.loadmat(data_dir + 'train_img_lab.mat')
            labels_train = train_img_lab['train_img_lab']
            test_img_lab = sio.loadmat(data_dir + 'test_img_lab.mat')
            labels_test = test_img_lab['test_img_lab']

            # from sklearn import preprocessing
            # img_scaler = preprocessing.StandardScaler().fit(I_train)
            # I_train = img_scaler.transform(I_train)
            # I_test = img_scaler.transform(I_test)
            #
            # txt_scaler = preprocessing.StandardScaler().fit(T_train)
            # T_train = txt_scaler.transform(T_train)
            # T_test = txt_scaler.transform(T_test)

            # sio.savemat('/root/workspace/datasets/XMediaNet/my_extracted_feature/xmedianet_features.mat',
            #             {'X1': I_train,
            #              'X2': T_train,
            #              'XTe1': I_test,
            #              'XTe2': T_test,
            #              'trainLabel': labels_train,
            #              'testLabel': labels_test})

        elif dataset == 'xmedianet_av':
            import h5py
            data_dir = '/root/workspace/datasets/XMediaNet/my_extracted_feature/'
            data = h5py.File(data_dir + 'xmedianet_aud_deep_vid_data_has_valid.h5py')
            I_train = data['train_vids_deep'].value
            T_train = data['train_auds'].value

            I_test = data['test_vids_deep'].value
            T_test = data['test_auds'].value

            labels_train = data['train_vids_labels'].value - 1
            labels_test = data['test_auds_labels'].value - 1

            from sklearn import preprocessing
            img_scaler = preprocessing.MinMaxScaler().fit(I_train)
            I_train = img_scaler.transform(I_train)
            I_test = img_scaler.transform(I_test)

            txt_scaler = preprocessing.MinMaxScaler().fit(T_train)
            T_train = txt_scaler.transform(T_train)
            T_test = txt_scaler.transform(T_test)

        elif dataset == 'nus_deep':
            data_dir = '/root/workspace/datasets/nus/my_extracted_feature/'
            train_img = sio.loadmat(data_dir + 'train_img.mat')
            I_train = train_img['train_img']
            train_txt = sio.loadmat(data_dir + 'train_txt.mat')
            T_train = train_txt['train_txt']

            test_img = sio.loadmat(data_dir + 'test_img.mat')
            I_test = test_img['test_img']
            test_txt = sio.loadmat(data_dir + 'test_txt.mat')
            T_test = test_txt['test_txt']

            train_img_lab = sio.loadmat(data_dir + 'train_img_lab.mat')
            labels_train = train_img_lab['train_img_lab']
            test_img_lab = sio.loadmat(data_dir + 'test_img_lab.mat')
            labels_test = test_img_lab['test_img_lab']

            from sklearn import preprocessing
            img_scaler = preprocessing.StandardScaler().fit(I_train)
            I_train = img_scaler.transform(I_train)
            I_test = img_scaler.transform(I_test)

            txt_scaler = preprocessing.StandardScaler().fit(T_train)
            T_train = txt_scaler.transform(T_train)
            T_test = txt_scaler.transform(T_test)

            # sio.savemat('/root/workspace/datasets/nus/my_extracted_feature/nus_features.mat',
            #             {'X1': I_train,
            #              'X2': T_train,
            #              'XTe1': I_test,
            #              'XTe2': T_test,
            #              'trainLabel': labels_train,
            #              'testLabel': labels_test})
        elif dataset == 'half_mnist':
            data_dir = '/root/workspace/datasets/half_mnist/'
            data = sio.loadmat(data_dir + 'half_MNIST.mat')
            I_train = data['X1']
            T_train = data['X2']

            I_test = data['XTe1']
            T_test = data['XTe2']


            labels_train = data['trainLabel']
            labels_test = data['testLabel']

            # from sklearn import preprocessing
            # img_scaler = preprocessing.StandardScaler().fit(I_train)
            # I_train = img_scaler.transform(I_train)
            # I_test = img_scaler.transform(I_test)
            #
            # txt_scaler = preprocessing.StandardScaler().fit(T_train)
            # T_train = txt_scaler.transform(T_train)
            # T_test = txt_scaler.transform(T_test)

        if state == 'train':
            self.I = I_train
            self.T = T_train
            self.labels = labels_train

        if state == 'test':
            self.I = I_test
            self.T = T_test
            self.labels = labels_test

        self.I = torch.FloatTensor(self.I)

        if dataset == 'wiki_shallow' or dataset == 'wiki_deep' or dataset == 'pascal_deep' \
                or dataset == 'xmedianet_deep' or dataset == 'nus_deep' or dataset == 'half_mnist' \
                or dataset == 'xmedianet_av':
            self.T = torch.FloatTensor(self.T)
        elif dataset == 'xmedianet' or dataset == 'nus_deep' or dataset == 'wiki_deep_corr-ae':
            self.T = torch.LongTensor(self.T)

        self.labels = torch.LongTensor(self.labels)
        self.labels = self.labels.view(-1, 1)

    def __getitem__(self, index):
        I_item, T_item, label = self.I[index], self.T[index], self.labels[index]
        return I_item, T_item, label

    def __len__(self):
        count = len(self.I)
        # print (len(self.I), len(self.T), len(self.labels))
        assert len(self.I) == len(self.T) == len(self.labels)
        return count
