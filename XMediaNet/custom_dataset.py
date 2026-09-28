from torchvision.datasets.folder import default_loader
import os
from torch.utils.data.dataset import Dataset
import torch
import scipy.io as sio
import h5py
import numpy as np
from sklearn.decomposition import PCA
import random


class MyCustomDataset(Dataset):
    def __init__(self, dataset_name='wiki_shallow', state='train', gamma=0.5):
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


        elif dataset_name == 'wiki_shallow':
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
        elif dataset_name == 'xmedianet':
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

        elif dataset_name == 'nus':
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

        elif dataset_name == 'wiki_deep_corr-ae':
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

        elif dataset_name == 'wiki_deep':
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

        elif dataset_name == 'my_wiki_deep':
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

        elif dataset_name == 'pascal_deep':
            if gamma != 0:
                data_dir = '/root/workspace/dataset/pascal/data_split_' + str(gamma) + '/'
                lab_train_img = sio.loadmat(data_dir + 'train_img.mat')
                lab_I_train = lab_train_img['train_img']
                unlab_train_img = sio.loadmat(data_dir + 'unlab_train_img.mat')
                unlab_I_train = unlab_train_img['unlab_train_img']
                I_train = np.concatenate((lab_I_train, unlab_I_train), 0)

                lab_train_txt = sio.loadmat(data_dir + 'lab_train_txt.mat')
                lab_T_train = lab_train_txt['lab_train_txt']
                unlab_train_txt = sio.loadmat(data_dir + 'unlab_train_txt.mat')
                unlab_T_train = unlab_train_txt['unlab_train_txt']
                T_train = np.concatenate((lab_T_train, unlab_T_train), 0)

                test_img = sio.loadmat(data_dir + 'test_img.mat')
                I_test = test_img['test_img']
                test_txt = sio.loadmat(data_dir + 'test_txt.mat')
                T_test = test_txt['test_txt']

                lab_train_imgs_label = sio.loadmat(data_dir + 'train_img_lab.mat')
                lab_train_imgs_label = lab_train_imgs_label['train_img_lab']
                unlab_train_imgs_label = sio.loadmat(data_dir + 'unlab_train_img_lab.mat')
                unlab_train_imgs_label = unlab_train_imgs_label['unlab_train_img_lab']
                labels_train = np.concatenate((lab_train_imgs_label, unlab_train_imgs_label), 0)

                test_imgs_label = sio.loadmat(data_dir + 'test_img_lab.mat')
                labels_test = test_imgs_label['test_img_lab']

            if gamma == 0:
                data_dir = '/root/workspace/dataset/pascal/my_extracted_feature/'
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

        elif dataset_name == 'xmedianet_deep':
            num_classes = 200

            # data_dir = '/root/workspace/datasets/XMediaNet/my_extracted_feature/data_split_' + str(gamma) + '/'
            data_dir = '/root/workspace/img_txt_pretrain/xmedianet/CNN_text_xmedianet_ss_split/data_split_' + str(gamma) + '/'
            lab_train_img = sio.loadmat(data_dir + 'lab_train_imgs.mat')
            lab_I_train = lab_train_img['lab_train_imgs']
            unlab_train_img = sio.loadmat(data_dir + 'unlab_train_imgs.mat')
            unlab_I_train = unlab_train_img['unlab_train_imgs']
            I_train = np.concatenate((lab_I_train, unlab_I_train), 0)

            lab_train_txt = sio.loadmat(data_dir + 'lab_train_txt.mat')
            lab_T_train = lab_train_txt['lab_train_txt']
            unlab_train_txt = sio.loadmat(data_dir + 'unlab_train_txt.mat')
            unlab_T_train = unlab_train_txt['unlab_train_txt']
            T_train = np.concatenate((lab_T_train, unlab_T_train), 0)

            test_img = sio.loadmat(data_dir + 'test_imgs.mat')
            I_test = test_img['test_imgs']
            test_txt = sio.loadmat(data_dir + 'test_txt.mat')
            T_test = test_txt['test_txt']

            lab_train_imgs_label = sio.loadmat(data_dir + 'lab_train_imgs_label.mat')
            lab_train_imgs_label = lab_train_imgs_label['lab_train_imgs_label'].T
            unlab_train_imgs_label = sio.loadmat(data_dir + 'unlab_train_imgs_label.mat')
            unlab_train_imgs_label = unlab_train_imgs_label['unlab_train_imgs_label'].T
            labels_train = np.concatenate((lab_train_imgs_label, unlab_train_imgs_label), 0)

            test_imgs_label = sio.loadmat(data_dir + 'test_imgs_label.mat')
            labels_test = test_imgs_label['test_imgs_label']
            print()


            # data_dir = '/root/workspace/datasets/XMediaNet/my_extracted_feature/'
            # train_img = sio.loadmat(data_dir + 'train_img.mat')
            # I_train = train_img['train_img']
            # train_txt = sio.loadmat(data_dir + 'train_txt.mat')
            # T_train = train_txt['train_txt']
            #
            # test_img = sio.loadmat(data_dir + 'test_img.mat')
            # I_test = test_img['test_img']
            # test_txt = sio.loadmat(data_dir + 'test_txt.mat')
            # T_test = test_txt['test_txt']
            #
            # train_img_lab = sio.loadmat(data_dir + 'train_img_lab.mat')
            # labels_train = train_img_lab['train_img_lab']
            # test_img_lab = sio.loadmat(data_dir + 'test_img_lab.mat')
            # labels_test = test_img_lab['test_img_lab']
            # print()


            # import h5py
            # data_dir = '/root/workspace/datasets/XMediaNet/my_extracted_feature/hupeng/'
            # data = h5py.File(data_dir + 'xmedianet_deep_doc2vec_data.h5py', 'r')
            # I_train = data['train_imgs_deep'][()].astype('float32')
            # T_train = data['train_text'][()].astype('float32')
            #
            #
            # I_test = data['test_imgs_deep'][()].astype('float32')
            # T_test = data['test_text'][()].astype('float32')
            #
            # labels_train = data['train_imgs_labels'][()]
            # labels_train -= np.min(labels_train)
            # labels_test = data['test_imgs_labels'][()]
            # labels_test -= np.min(labels_test)

            # data_dir = '/root/workspace/datasets/XMediaNet/my_extracted_feature/hupeng/'
            # train_img = sio.loadmat(data_dir + 'I_rbm_train.mat')
            # I_train = train_img['I_rbm_train']
            # train_txt = sio.loadmat(data_dir + 'T_rbm_train.mat')
            # T_train = train_txt['T_rbm_train']
            #
            # test_img = sio.loadmat(data_dir + 'I_rbm_test.mat')
            # I_test = test_img['I_rbm_test']
            # test_txt = sio.loadmat(data_dir + 'T_rbm_test.mat')
            # T_test = test_txt['T_rbm_test']
            #
            # train_img_lab = sio.loadmat(data_dir + 'labels_train.mat')
            # labels_train = train_img_lab['labels_train'].T
            # test_img_lab = sio.loadmat(data_dir + 'labels_test.mat')
            # labels_test = test_img_lab['labels_test'].T


            test_ind = []
            # for i in range(num_classes):
            #     ind_i = np.where(labels_test == i)[0]
            #     test_ind_i = random.sample(list(ind_i), int(len(ind_i) * 0.5))
            #     test_ind.extend(test_ind_i)
            # test_ind = range(0, int(len(labels_test)*0.5))
            # I_test = I_test[test_ind]
            # T_test = T_test[test_ind]
            # labels_test = labels_test[test_ind]

            ### Semi-Supervised Setting

            # for i in range(num_classes):
            #     ind_i = np.where(labels_train == i)[0]
            #     ulab_ind_i = random.sample(list(ind_i), int(len(ind_i)*gamma))
            #     lab_ind_i = [i for i in list(ind_i) if i not in ulab_ind_i]
            #     labels_train[ulab_ind_i] = -1


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

        elif dataset_name == 'xmedianet_av':
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

        elif dataset_name == 'nus_deep':
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
        elif dataset_name == 'half_mnist':
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

        if dataset_name == 'wiki_shallow' or dataset_name == 'wiki_deep' or dataset_name == 'pascal_deep' \
                or dataset_name == 'xmedianet_deep' or dataset_name == 'nus_deep' or dataset_name == 'half_mnist' \
                or dataset_name == 'xmedianet_av':
            self.T = torch.FloatTensor(self.T)
        elif dataset_name == 'xmedianet' or dataset_name == 'nus_deep' or dataset_name == 'wiki_deep_corr-ae':
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
