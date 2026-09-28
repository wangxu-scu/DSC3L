import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


class TextCNN(nn.Module):
    def __init__(
            self,
            in_channels,
            kernel_number,
            voca_size,
            embed_dim,
            kernel_sizes,
            dropout_p,
            embedding_weight,
            outpt_dim):
        super(TextCNN, self).__init__()
        V = voca_size
        D = embed_dim
        Ci = in_channels
        Co = kernel_number
        Ks = kernel_sizes
        self.embedding_weight = embedding_weight

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(len(Ks) * Co, outpt_dim)
        self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def init_weights(self):
        self.embed.weight = nn.Parameter(self.embedding_weight)
        # self.fc.bias.data.normal_(0, 0.01)
        # self.fc.weight.data.normal_(0, 0.01)
        #
        # for layer in self.convs:
        #     nn.init.xavier_normal(layer.weight)

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        # if self.args.static:  ## fix the embedding
        #     x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3)
             for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        # x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.fc1(x), 0.01)
        # x = self.dropout(x)  # (N, len(Ks)*Co)
        # logit = self.fc1(x)  # (N, C)
        return x



class ImageCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImageCNN, self).__init__()
        self.Sequential = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Dropout(0.5),
                                        nn.LeakyReLU(0.01),
                                        nn.Linear(hidden_dim, output_dim),
                                        nn.Dropout(0.5),
                                        nn.LeakyReLU(0.01)
                                        )
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x = F.leaky_relu(self.fc1(x), 0.05)
        # x = F.leaky_relu(self.fc2(x), 0.05)
        x = self.Sequential(x)
        return x


class CommonDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CommonDNN, self).__init__()
        self.Sequential = nn.Sequential(nn.Linear(input_dim, output_dim)
                                        # nn.Dropout(0.5),
                                        # nn.LeakyReLU(),
                                        # nn.Linear(hidden_dim, output_dim)
                                        )
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x = F.leaky_relu(self.fc1(x), 0.05)
        # x = self.fc2(x)
        x = self.Sequential(x)
        return x


class CNN(nn.Module):
    def __init__(
            self,
            input_dim_I,
            embedding_weight,
            hidden_dim1,
            hidden_dim2,
            output_dim):
        super(CNN, self).__init__()
        in_channels = 1
        kernel_number = 100
        voca_size = embedding_weight.shape[0]
        embed_dim = embedding_weight.shape[1]
        kernel_sizes = [3, 4, 5]
        dropout_p = 0.5
        self.TextCNN = TextCNN(
            in_channels,
            kernel_number,
            voca_size,
            embed_dim,
            kernel_sizes,
            dropout_p,
            embedding_weight,
            hidden_dim1)

        self.ImageCNN = ImageCNN(input_dim_I, hidden_dim1, hidden_dim1)
        self.CommonDNN = CommonDNN(hidden_dim1, hidden_dim2, output_dim)


    def forward(self, img, text):
        # Image Pathway
        y_I = self.ImageCNN(img)
        y_I_C = self.CommonDNN(y_I)
        # y_I_C = y_I

        # Text Pathway
        y_T = self.TextCNN(text)
        y_T_C = self.CommonDNN(y_T)
        # y_T_C = y_T
        return y_I, y_T, y_I_C, y_T_C
