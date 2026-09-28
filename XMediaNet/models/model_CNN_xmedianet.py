import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


class TextCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextCNN, self).__init__()
        self.Sequential = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        # nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, output_dim),
                                        # nn.BatchNorm1d(output_dim),
                                        nn.ReLU(),
                                        # nn.Dropout(0.5),
                                        )

        # self.Sequential = nn.Sequential(nn.Linear(input_dim, output_dim),
        #                                 # nn.BatchNorm1d(hidden_dim),
        #                                 nn.ReLU()
        #                                 )
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = F.leaky_relu(self.fc1(x), 0.05)
        # x = F.leaky_relu(self.fc2(x), 0.05)
        x = self.Sequential(x)

        return x


class ImageCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImageCNN, self).__init__()
        self.Sequential = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        # nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, output_dim),
                                        # nn.BatchNorm1d(output_dim),
                                        nn.ReLU(),
                                        # nn.Dropout(0.5),
                                        )

        # self.Sequential = nn.Sequential(nn.Linear(input_dim, output_dim),
        #                                 # nn.BatchNorm1d(hidden_dim),
        #                                 nn.ReLU()
        #                                 )
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = F.leaky_relu(self.fc1(x), 0.05)
        # x = F.leaky_relu(self.fc2(x), 0.05)
        x = self.Sequential(x)

        return x


# class CommonDNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(CommonDNN, self).__init__()
#         self.Sequential = nn.Sequential(nn.Linear(input_dim, output_dim),
#                                         nn.Dropout(0.5),
#                                         nn.LeakyReLU(0.01),
#                                         # nn.Linear(hidden_dim, output_dim)
#                                         )
#
#         # self.Sequential = nn.Sequential(nn.Linear(input_dim, hidden_dim),
#         #                                 nn.BatchNorm1d(hidden_dim),
#         #                                 nn.ReLU(),
#         #                                 nn.Linear(hidden_dim, output_dim)
#         #                                 )
#
#         # self.fc1 = nn.Linear(input_dim, hidden_dim)
#         # self.fc2 = nn.Linear(hidden_dim, output_dim)
#     def forward(self, x):
#         # x = F.leaky_relu(self.fc1(x), 0.05)
#         # x = self.fc2(x)
#         x = self.Sequential(x)
#         return x


class CNN(nn.Module):
    def __init__(
            self,
            input_dim_I,
            input_dim_T,
            hidden_dim,
            output_dim):
        super(CNN, self).__init__()

        self.TextCNN = TextCNN(input_dim_T, hidden_dim, output_dim)

        # mode = 'multichannel'
        # word_dim = embedding_weight.shape[1]
        # vocab_size = embedding_weight.shape[0]
        # filters = [3, 4, 5]
        # filter_num = [100, 100, 100]
        # dropout_prob = 0.5
        # wv_matrix = embedding_weight
        #
        # self.TextCNN = TextCNN(mode,
        #                        word_dim,
        #                        vocab_size,
        #                        hidden_dim1,
        #                        filters,
        #                        filter_num,
        #                        dropout_prob,
        #                        wv_matrix)

        self.ImageCNN = ImageCNN(input_dim_I, hidden_dim, output_dim)
        # self.ImageCNN = Image_CNN_list(input_dim_I, output_dim)
        # self.CommonDNN = CommonDNN(hidden_dim1, hidden_dim2, output_dim)

    def forward(self, img, text):
        # Image Pathway
        y_I = self.ImageCNN(img)
        # y_I = self.CommonDNN(y_I)
        # y_I = y_I / torch.norm(y_I, dim=1, keepdim=True)

        # Text Pathway
        y_T = self.TextCNN(text)
        # y_T = self.CommonDNN(y_T)
        # y_T = y_T / torch.norm(y_T, dim=1, keepdim=True)
        return y_I, y_T
