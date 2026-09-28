import h5py
data = h5py.File('XMediaNet/xmedianet_deep_idx_data.h5py', 'r')
I_train = data['train_imgs_deep']
I_train_labels = data['train_imgs_labels']
T_train= data['train_texts_idx']
T_train_labels = data['train_texts_labels']
I_test = data['test_imgs_deep']
I_test_labels =data['test_imgs_labels']
T_test = data['test_texts_idx']
T_test_labels =data['test_texts_labels']
wv_matrix = data['wv_matrix']