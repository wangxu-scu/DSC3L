import time
import copy
import torch
import torch.nn as nn
import numpy as np
# import scipy
from numpy.matlib import repmat
from fxeval import fx_calc_map_label, fx_calc_map_nolabel_top
from mcml_loss import comp_cost
import utils

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



def train(
        img_model,
        text_model,
        dataloaders,
        device,
        dataset_sizes,
        num_epochs):
    since = time.time()

    best_img_model_wts = copy.deepcopy(img_model.state_dict())
    best_txt_model_wts = copy.deepcopy(text_model.state_dict())

    # optimizer
    img_optimizer = torch.optim.Adam(img_model.parameters(), lr=0.001)
    txt_optimizer = torch.optim.Adam(text_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # scheduler.step()
                img_model.train()  # Set model to training mode
                text_model.train()
            else:
                img_model.eval()   # Set model to evaluate mode
                text_model.eval()
            running_loss = 0.0

            # Iterate over data.
            for i, batch in enumerate(dataloaders[phase]):
                imgs, texts, labels = batch[0], batch[1], batch[2]
                # print (imgs.shape)
                # print(txts.shape)
                imgs = imgs.to(device)
                texts = texts.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                img_optimizer.zero_grad()
                txt_optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    img_feas = img_model(imgs)
                    text_feas = text_model(texts)
                    loss = comp_cost(img_feas, text_feas, labels)
                    loss = loss.to(device)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        img_optimizer.step()
                        loss.backward(retain_graph=True)
                        txt_optimizer.step()

                # statistics
                running_loss += loss.item() * imgs.size(0)
                print('{} running_loss: {:.4f}'.format(phase, running_loss
                                                       ))

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # # deep copy the model
            # if phase == 'test' and epoch_img_acc > best_img_acc:
            #     best_img_acc = epoch_img_acc
            #     best_img_model_wts = copy.deepcopy(img_model.state_dict())
            # if phase == 'test' and epoch_txt_acc > best_txt_acc:
            #     best_txt_acc = epoch_txt_acc
            #     best_txt_model_wts = copy.deepcopy(txt_model.state_dict())

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
    return img_model, text_model


def train2(model,
           dataloaders,
           device,
           dataset_sizes,
           num_epochs):
    since = time.time()

    # optimizer
    # optimizer = torch.optim.LBFGS(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-6)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    best_mean_map = 0.
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:

            num_batch = int(np.ceil(dataset_sizes[phase] / dataloaders[phase].batch_size))

            if phase == 'train':
                # scheduler.step()
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
                    img_feas, text_feas, H_I, H_T = model(imgs, texts)

                    # feas = torch.cat((img_feas, text_feas), 0)
                    # labels = torch.cat((labels, labels), 0)
                    # loss = nn.CrossEntropyLoss()(feas, labels.squeeze(1))

                    # img_loss = nn.CrossEntropyLoss()(img_feas, labels.squeeze(1))
                    # text_loss = nn.CrossEntropyLoss()(text_feas, labels.squeeze(1))
                    # loss = img_loss + text_loss

                    loss = comp_cost(img_feas, text_feas, labels, device)

                    # loss2 = ((H_I - H_T) ** 2).sum()
                    # loss = loss + 0.001 * loss2
                    loss = loss.to(device)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()

                        # # Obtain low-rank projection
                        # W = model.C_Sequential.weight
                        # q = W.size(0) ### The dim of the low dimensional data
                        # A = W.t() @ W
                        # E, V = torch.eig(A, eigenvectors=True)
                        # E = E[:, 0]
                        # E[E<0]=0
                        # E[q:]=0
                        # lambda_W = torch.diag(torch.sqrt(E[0:q]))
                        # V_W = V.t()[0:q, :]
                        # W = lambda_W @ V_W
                        # model.C_Sequential.weight = torch.nn.Parameter(W)

                # statistics
                current_loss = loss.item()
                running_loss += current_loss * imgs.size(0)
                # print('{} running_loss: {:.4f}'.format(phase, loss.item() * imgs.size(0)))
                utils.show_progressbar([i, num_batch], loss=current_loss)

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        if (epoch + 1) % 10 ==0 or (epoch + 1) >= 2:
            img_feas_list, text_feas_list, label_list = [], [], []
            for i, batch in enumerate(dataloaders['test']):
                imgs, texts, labels = batch[0], batch[1], batch[2]
                imgs = imgs.to(device)
                texts = texts.to(device)
                img_feas, text_feas, H_I_feas, H_T_feas = model(imgs, texts)
                img_feas_list.append(img_feas.cpu().detach().numpy())
                text_feas_list.append(text_feas.cpu().detach().numpy())
                label_list.append(labels)

            img_feas = np.concatenate(img_feas_list, 0)
            text_feas = np.concatenate(text_feas_list, 0)
            labels = np.concatenate(label_list, 0)

            # img_feas = znorm(img_feas)
            # text_feas = znorm(text_feas)
            result1_all = fx_calc_map_label(
                img_feas, text_feas, labels, k=0, dist_method='COS')
            result2_all = fx_calc_map_label(
                text_feas, img_feas, labels, k=0, dist_method='COS')
            result1_50 = fx_calc_map_label(
                img_feas, text_feas, labels, k=100, dist_method='COS')
            result2_50 = fx_calc_map_label(
                text_feas, img_feas, labels, k=100, dist_method='COS')

            print("Image query, MAP(R@All): {}".format(result1_all))
            print("Text query, MAP(R@All): {}".format(result2_all))
            print("Image query, MAP(R@100): {}".format(result1_50))
            print("Text query, MAP(R@100): {}".format(result2_50))

            mean_map = (result1_all + result2_all) / 2.0
            if mean_map > best_mean_map:
                best_mean_map = mean_map
                print('best map improves to: {}, detail: i2t: {}, t2i: {}'.format(best_mean_map, result1_all, result2_all))
            else:
                print('best map still is: {}'.format(best_mean_map))

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
  col=inMat.shape[0]
  row=inMat.shape[1]
  mean_val=np.mean(inMat, axis=0)
  std_val=np.std(inMat, axis=0)
  mean_val=repmat(mean_val, col, 1)
  std_val=repmat(std_val, col, 1)
  x = np.argwhere(std_val==0)
  for y in x:
    std_val[y[0],y[1]]=1
  return (inMat-mean_val)/std_val