import time
import copy
import torch
import torch.nn as nn
import numpy as np
# import scipy
from numpy.matlib import repmat
from custom_dataset import MyCustomDataset
from torch.utils.data import DataLoader
from loss import cmpm_loss_compute, mcml_loss_compute, xentropy_loss_compute, pairwise_KL
from utils import show_progressbar, cal_map_bi, cal_map_all, cal_knn_acc
from models.model_CNN_pascal import CNN



# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")




# Training settings
class Solver(object):
    def __init__(self, args):
        print('dataset loading')
        self.dataset, self.dataloaders, self.dataset_sizes = self.load_data(args)
        print('load finished!')

        print('set device')
        self.device = self.set_device(args.device)
        print('build model')
        self.model = CNN(input_dim_I=args.input_dim_I, input_dim_T=args.input_dim_T,
                         hidden_dim=args.hidden_dim, output_dim=args.output_dim)
        self.model.to(self.device)
        self.opt = self.set_optimizer(args)


    def set_optimizer(self, args):
        if args.optimizer == 'momentum':
            opt = torch.optim.SGD(self.model.parameters(),
                                   lr=args.lr, weight_decay=0.0005,
                                   momentum=args.momentum)
        elif args.optimizer == 'adam':
            opt = torch.optim.Adam(self.model.parameters(),
                                    lr=args.lr, betas=(0.5, 0.999))
        else:
            return None
        return opt

    def load_data(self, args):
        dataset = {x: MyCustomDataset(dataset_name=args.dataset_name, state=x, gamma=args.gamma)
                   for x in ['train', 'test']}
        dataloaders = {x: DataLoader(dataset[x], batch_size=args.batch_size,
                                     shuffle=True, num_workers=0)
                       for x in ['train', 'test']}
        dataset_sizes = {x: len(dataset[x]) for x in ['train', 'test']}
        return dataset, dataloaders, dataset_sizes

    def set_device(self, device):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        return device

    def cal_loss(self, img_feas, text_feas, labels, args, phase='train'):
        ## Supervised Term
        sup_ind = np.where(labels.cpu() != -1)[0]
        sup_labels = labels[sup_ind, :]
        sup_img_feas = img_feas[sup_ind, :]
        sup_text_feas = text_feas[sup_ind, :]

        sup_loss = 0
        if sup_img_feas.shape[0] > 0:
            if args.loss_type == 'cmpm':
                sup_loss = cmpm_loss_compute(sup_img_feas, sup_text_feas, sup_labels)
            elif args.loss_type == 'mcml':
                sup_loss = mcml_loss_compute(sup_img_feas, sup_text_feas, sup_labels, args)
            elif args.loss_type == 'xentropy':
                sup_loss = xentropy_loss_compute(sup_img_feas, sup_text_feas, sup_labels)
            else:
                raise ('wrong loss type')

        ## Unsupervised Term
        if args.gamma !=0 and phase == 'train':
            unsup_ind = np.where(labels.cpu() == -1)[0]
            unsup_img_feas = img_feas[unsup_ind, :]
            unsup_text_feas = text_feas[unsup_ind, :]

            pairwse_loss = pairwise_KL(unsup_img_feas, unsup_text_feas)
        else:
            pairwse_loss = 0

        ## Total Loss
        loss = args.alpha * sup_loss + args.beta * pairwse_loss
        return loss

    def train(self, args, retreival=True):
        since = time.time()
        # grad_params = [param for param in model.parameters()
        #                if param.requires_grad]
        # optimizer = torch.optim.Adam(
        #     grad_params,
        #     lr=1e-4,
        #     betas=(
        #         0.5,
        #         0.99),
        #     weight_decay=1e-4)
        # optimizer = torch.optim.SGD(grad_params, lr=0.001, momentum=0.9, weight_decay=0.001)
        # optimizer = torch.optim.ASGD(model.parameters(), lr=1e-1)

        if retreival is True:
            best_map_bi_50, best_map_bi_all = 0., 0.
            best_map_all_50, best_map_all_all = 0., 0.
        else:
            best_acc = 0.
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.opt, step_size=1, gamma=0.98)
        pascal_loss_history = []
        pascal_map_history = []
        for epoch in range(args.max_epoch):
            # scheduler.step()
            # print('-' * 10)
            print('Epoch {}/{}'.format(epoch + 1, args.max_epoch))
            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:

                num_batch = int(np.ceil(self.dataset_sizes[phase] /
                        self.dataloaders[phase].batch_size))

                if phase == 'train':
                    # scheduler.step()
                    self.model.train()  # Set model to training mode

                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                for i, batch in enumerate(self.dataloaders[phase]):
                    imgs, texts, labels = batch[0], batch[1], batch[2]
                    imgs = imgs.to(self.device)
                    texts = texts.to(self.device)
                    labels = labels.to(self.device)
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # # zero the parameter gradients
                        self.opt.zero_grad()
                        img_feas, text_feas = self.model(imgs, texts)

                        ### ==== loss ==== ###
                        loss = self.cal_loss(img_feas, text_feas, labels, args, phase)
                        # loss2 = torch.mean(((img_feas - text_feas) ** 2).sum(1)) / 2.0
                        # loss = loss + 0.001 * loss2
                        # reg_loss = 0
                        # for param in self.model.parameters():
                        #     reg_loss += param.norm(2)
                        # l1_factor = 0.0000
                        # loss +=  l1_factor * reg_loss
                        loss = loss.to(self.device)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            # loss.backward(retain_graph=True)
                            loss.backward()
                            # com_optimizer.step()
                            self.opt.step()

                    # statistics
                    current_loss = loss.item()
                    running_loss += current_loss * imgs.size(0)
                    # print('{} running_loss: {:.4f}'.format(phase, loss.item() * imgs.size(0)))
                    show_progressbar([i, num_batch], loss=current_loss)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                if phase == 'train':
                    pascal_loss_history.append(epoch_loss)
                import scipy.io as sio
                sio.savemat('pascal_loss_history.mat', {'pascal_loss_history': pascal_loss_history})

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # if (epoch + 1) % 10 == 0 or (epoch + 1) >= 1:
            if (epoch + 1) >= args.test_epoch:
                img_feas_list, text_feas_list, label_list = [], [], []
                for i, batch in enumerate(self.dataloaders['test']):
                    imgs, texts, labels = batch[0], batch[1], batch[2]
                    imgs = imgs.to(self.device)
                    texts = texts.to(self.device)
                    img_feas, text_feas = self.model(imgs, texts)
                    img_feas_list.append(img_feas.cpu().detach().numpy())
                    text_feas_list.append(text_feas.cpu().detach().numpy())
                    label_list.append(labels)

                img_feas = np.concatenate(img_feas_list, 0)
                text_feas = np.concatenate(text_feas_list, 0)
                labels = np.concatenate(label_list, 0)

                # img_feas = znorm(img_feas)
                # text_feas = znorm(text_feas)

                if retreival is True:
                    # best_map_bi_50, best_map_bi_all = \
                    #     cal_map_bi(img_feas, text_feas, labels, best_map_bi_50, best_map_bi_all, dataset_name=args.dataset_name)

                    best_map_bi_50, best_map_bi_all, is_best, result1_all, result2_all = cal_map_bi(img_feas, text_feas, labels, best_map_bi_50, best_map_bi_all, dataset_name=args.dataset_name, get_all=True)
                    if is_best:
                        best_img2txt, best_txt2img = result1_all, result2_all
                    best_map_all_50, best_map_all_all = \
                        cal_map_all(img_feas, text_feas, labels, best_map_all_50, best_map_all_all)
                    pascal_map_history.append(best_map_bi_all)
                    sio.savemat('pascal_map_history.mat', {'pascal_map_history': pascal_map_history})
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
        # return self.model
        return best_img2txt, best_txt2img


    def znorm(self, inMat):
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
