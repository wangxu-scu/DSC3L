import time
import copy
import torch
import torch.nn as nn
import numpy as np
# import scipy
from numpy.matlib import repmat
from fxeval import fx_calc_map_label, fx_calc_map_nolabel_top
from loss import cmpm_loss_compute, mcml_loss_compute, xentropy_loss_compute

from utils import show_progressbar, cal_map_bi, cal_map_all, cal_knn_acc

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset_name='xmedianet'
def train2(model,
           dataloaders,
           device,
           dataset_sizes,
           loss_type,
           num_epochs,
           retreival=True):
    since = time.time()

    # optimizer
    # optimizer = torch.optim.LBFGS(model.parameters())

    # com_params = [model.CommonDNN.Sequential[0].weight]
    grad_params = [param for param in model.parameters()
                   if param.requires_grad]
    optimizer = torch.optim.Adam(
        grad_params,
        lr=1e-4,
        betas=(
            0.5,
            0.99),
        weight_decay=1e-4)
    # optimizer = torch.optim.SGD(grad_para
    # ms, lr=0.001, momentum=0.9, weight_decay=0.001)
    # optimizer = torch.optim.ASGD(model.parameters(), lr=1e-1)

    if retreival is True:
        best_map_bi_50, best_map_bi_all = 0., 0.
        best_map_all_50, best_map_all_all = 0., 0.
    else:
        best_acc = 0.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.98)
    xmedianet_loss_history = []
    xmedianet_map_history = []
    for epoch in range(num_epochs):
        # scheduler.step()
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:

            num_batch = int(
                np.ceil(
                    dataset_sizes[phase] /
                    dataloaders[phase].batch_size))

            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode

            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, batch in enumerate(dataloaders[phase]):
                imgs, texts, labels = batch[0], batch[1], batch[2]
                imgs = imgs.to(device)
                texts = texts.to(device)
                labels = labels.to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # # zero the parameter gradients
                    optimizer.zero_grad()
                    img_feas, text_feas = model(imgs, texts)

                    ### ==== loss ==== ###
                    if loss_type == 'cmpm':
                        loss = cmpm_loss_compute(img_feas, text_feas, labels)
                    elif loss_type == 'mcml':
                        loss = mcml_loss_compute(img_feas, text_feas, labels)
                    elif loss_type == 'xentropy':
                        loss = xentropy_loss_compute(img_feas, text_feas, labels)
                    else:
                        raise ('wrong loss type')
                    loss2 = torch.mean(((img_feas - text_feas) ** 2).sum(1)) / 2.0
                    # loss = loss + 0.001 * loss2
                    reg_loss = 0
                    for param in model.parameters():
                        reg_loss += param.norm(2)
                    l1_factor = 0.0000
                    loss +=  l1_factor * reg_loss
                    loss = loss.to(device)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # loss.backward(retain_graph=True)
                        loss.backward()
                        # com_optimizer.step()
                        optimizer.step()

                        # # Obtain low-rank projection
                        # W = model.CommonDNN.Sequential[0].weight
                        # q = W.size(0) ### The dim of the low dimensional data
                        # A = W.t() @ W
                        # E, V = torch.eig(A, eigenvectors=True)
                        # E = E[:, 0]
                        # E[E<0]=0
                        # E[q:]=0
                        # lambda_W = torch.diag(torch.sqrt(E[0:q]))
                        # V_W = V.t()[0:q, :]
                        # W = lambda_W @ V_W
                        # model.CommonDNN.Sequential[0].weight = torch.nn.Parameter(W)

                # statistics
                current_loss = loss.item()
                running_loss += current_loss * imgs.size(0)
                # print('{} running_loss: {:.4f}'.format(phase, loss.item() * imgs.size(0)))
                show_progressbar([i, num_batch], loss=current_loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                xmedianet_loss_history.append(epoch_loss)
            import scipy.io as sio
            sio.savemat('xmedianet_loss_history.mat', {'xmedianet_loss_history': xmedianet_loss_history})

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        # if (epoch + 1) % 10 == 0 or (epoch + 1) >= 1:
        if (epoch + 1) >= 1:
            img_feas_list, text_feas_list, label_list = [], [], []
            for i, batch in enumerate(dataloaders['test']):
                imgs, texts, labels = batch[0], batch[1], batch[2]
                imgs = imgs.to(device)
                texts = texts.to(device)
                img_feas, text_feas = model(imgs, texts)
                img_feas_list.append(img_feas.cpu().detach().numpy())
                text_feas_list.append(text_feas.cpu().detach().numpy())
                label_list.append(labels)

            img_feas = np.concatenate(img_feas_list, 0)
            text_feas = np.concatenate(text_feas_list, 0)
            labels = np.concatenate(label_list, 0)

            # img_feas = znorm(img_feas)
            # text_feas = znorm(text_feas)

            if retreival is True:
                best_map_bi_50, best_map_bi_all = \
                    cal_map_bi(img_feas, text_feas, labels, best_map_bi_50, best_map_bi_all, dataset_name=dataset_name)
                # best_map_all_50, best_map_all_all = \
                #     cal_map_all(img_feas, text_feas, labels, best_map_all_50, best_map_all_all)
                xmedianet_map_history.append(best_map_bi_all)
                sio.savemat('xmedianet_map_history.mat', {'xmedianet_map_history': xmedianet_map_history})
            else:
                best_acc = cal_knn_acc(img_feas, labels, text_feas, labels, best_acc)
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    # print(
    #     'Best val Acc: Img: {:4f}, Txt: {:4f}'.format(
    #         best_img_acc,
    #         best_txt_acc))

    # load best model weights
    # img_model.load_state_dict(best_img_model_wts)
    # txt_model.load_state_dict(best_txt_model_wts)
    return model


def znorm(inMat):
    col = inMat.shape[0]
    row = inMat.shape[1]
    mean_val = np.mean(inMat, axis=0)
    std_val = np.std(inMat, axis=0)
    mean_val = repmat(mean_val, col, 1)
    std_val = repmat(std_val, col, 1)
    x = np.argwhere(std_val == 0)
    for y in x:
        std_val[y[0], y[1]] = 1
    return (inMat - mean_val) / std_val
