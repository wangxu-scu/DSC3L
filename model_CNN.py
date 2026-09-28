import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


class TextCNN(nn.Module):
    def __init__(
            self,
            mode,
            in_channels,
            kernel_number,
            voca_size,
            embed_dim,
            kernel_sizes,
            dropout_p,
            embedding_weight,
            outpt_dim,
            class_number):
        super(TextCNN, self).__init__()
        self.in_channels = in_channels
        self.V = voca_size
        self.D = embed_dim
        self.mode = mode
        C = class_number
        Ci = in_channels
        Co = kernel_number
        Ks = kernel_sizes
        self.embedding_weight = embedding_weight

        self.embed = nn.Embedding(self.V, self.D)

        if self.mode == 'multichannel':
            self.embedding2 = nn.Embedding(self.V, self.D)
            Ci = 2
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(Ci, Co, (K, self.D)) for K in Ks])
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(len(Ks) * Co, outpt_dim)
        self.fc2 = nn.Linear(outpt_dim, outpt_dim)
        self.fc3 = nn.Linear(outpt_dim, C)
        self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def init_weights(self):
        self.embed.weight = nn.Parameter(self.embedding_weight)
        if self.mode == 'multichannel':
            self.embedding2.weight = nn.Parameter(self.embedding_weight)
        # self.fc.bias.data.normal_(0, 0.01)
        # self.fc.weight.data.normal_(0, 0.01)
        #
        # for layer in self.convs:
        #     nn.init.xavier_normal(layer.weight)

    def forward(self, x):
        out = self.embed(x)  # (N, W, D)
        # if self.args.static:  ## fix the embedding
        #     x = Variable(x)
        out = out.unsqueeze(1)  # (N, Ci, W, D)

        if self.mode == 'multichannel':
            out2 = self.embedding2(x).unsqueeze(1)
            out = torch.cat((out, out2), 1)

        x = [F.relu(conv(out)).squeeze(3)
             for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        # x = F.dropout(x, 0.5)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc2(x)
        x = F.relu(x)
        logit = self.fc3(x)  # (N, C)
        return x, logit

    # def __init__(self,
    #              mode,
    #              word_dim,
    #              vocab_size,
    #              out_dim,
    #              filters,
    #              filter_num,
    #              dropout_prob,
    #              wv_matrix,
    #              in_channel=1):
    #     super(TextCNN, self).__init__()
    #     self.mode = mode
    #     self.word_dim = word_dim
    #     self.vocab_size = vocab_size
    #     self.out_dim = out_dim
    #     self.filters = filters
    #     self.filter_num = filter_num
    #     self.dropout_prob = dropout_prob
    #     self.in_channel = in_channel
    #
    #     assert (len(self.filters) == len(self.filter_num))
    #     self.embedding = nn.Embedding(
    #         self.vocab_size,
    #         self.word_dim)
    #     if self.mode == "static" or self.mode == 'non-static' or self.mode == 'multichannel':
    #         self.wv_matrix = wv_matrix
    #         self.embedding.weight.data.copy_(self.wv_matrix)
    #         if self.mode == 'static':
    #             self.embedding.weight.requires_grad = False
    #         elif self.mode == 'multichannel':
    #             self.embedding2 = nn.Embedding(
    #                 self.vocab_size,
    #                 self.word_dim)
    #             self.embedding2.weight.data.copy_(self.wv_matrix)
    #             self.embedding2.weight.requires_grad = True
    #             self.in_channel = 2
    #
    #     self.convs1 = nn.ModuleList(
    #         [
    #             nn.Conv2d(
    #                 self.in_channel, out_channel, (K, self.word_dim)) for out_channel, K in zip(
    #                 self.filter_num, self.filters)])
    #     self.dropout = nn.Dropout(self.dropout_prob)
    #     # self.fc = nn.Linear(sum(self.filter_num), self.out_dim)
    #     self.fc1 = nn.Linear(sum(self.filter_num), out_dim)
    #     # self.fc2 = nn.Linear(1024, out_dim)
    #     # self.fc2 = nn.Linear(1024, out_dim)
    #
    # def forward(self, x):
    #     out = self.embedding(x).unsqueeze(1)
    #
    #     if self.mode == 'multichannel':
    #         out2 = self.embedding2(x).unsqueeze(1)
    #         out = torch.cat((out, out2), 1)
    #
    #     out = [F.leaky_relu(_conv(out)).squeeze(3) for _conv in self.convs1]
    #     out = [F.max_pool1d(o, o.size(2)).squeeze(2) for o in out]
    #     out = torch.cat(out, 1)
    #     # out = self.dropout(out)
    #     out = self.fc1(out)
    #     out = F.leaky_relu(out, 0.01)
    #     out = self.dropout(out)
    #     # out3 = self.fc2(out1)
    #     # return [out1, out3]
    #
    #     # out = self.fc2(out)
    #     return out


class ImageCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImageCNN, self).__init__()
        self.Sequential = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        # nn.Dropout(0.5),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, output_dim),
                                        # nn.Dropout(0.5),
                                        nn.ReLU()
                                        )

        # self.Sequential = nn.Sequential(nn.Linear(input_dim, hidden_dim),
        #                                 nn.BatchNorm1d(hidden_dim),
        #                                 nn.ReLU()
        #                                 )
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
        self.Sequential = nn.Sequential(nn.Linear(input_dim, output_dim),
                                        nn.Dropout(0.5),
                                        nn.LeakyReLU(0.01),
                                        # nn.Linear(hidden_dim, output_dim)
                                        )

        # self.Sequential = nn.Sequential(nn.Linear(input_dim, hidden_dim),
        #                                 nn.BatchNorm1d(hidden_dim),
        #                                 nn.ReLU(),
        #                                 nn.Linear(hidden_dim, output_dim)
        #                                 )

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
            output_dim,
            class_number,
            pretrained_wordcnn_path=None):
        super(CNN, self).__init__()
        mode = 'non-multichannel'
        in_channels = 1
        kernel_number = 100
        voca_size = embedding_weight.shape[0]
        embed_dim = embedding_weight.shape[1]
        kernel_sizes = [3, 4, 5]
        dropout_p = 0.5

        self.TextCNN = TextCNN(
            mode,
            in_channels,
            kernel_number,
            voca_size,
            embed_dim,
            kernel_sizes,
            dropout_p,
            embedding_weight,
            hidden_dim1,
            class_number)
        if pretrained_wordcnn_path is not None:
            self.TextCNN = torch.load(pretrained_wordcnn_path)
            for param in self.TextCNN.parameters():
                param.requires_grad = False

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

        self.ImageCNN = ImageCNN(input_dim_I, hidden_dim1, hidden_dim1)
        # self.ImageCNN = Image_CNN_list(input_dim_I, output_dim)
        self.CommonDNN = CommonDNN(hidden_dim1, hidden_dim2, output_dim)

    def forward(self, img, text):
        # Image Pathway
        y_I = self.ImageCNN(img)
        # y_I = self.CommonDNN(y_I)

        # Text Pathway
        y_T, _ = self.TextCNN(text)
        # y_T = self.CommonDNN(y_T)
        return y_I, y_T
