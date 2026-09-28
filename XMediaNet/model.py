import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


class TextDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextDNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # self.init_weights()

    def init_weights(self):
        self.embed.weight = nn.Parameter(self.embedding_weight)

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return y


class ImageDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImageDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return y



class DNN(nn.Module):
    def __init__(
            self,
            input_dim_I,
            input_dim_T,
            hidden_dim,
            output_dim):
        super(DNN, self).__init__()
        self.I_Sequential = nn.Sequential(nn.Linear(input_dim_I, hidden_dim),
                                          nn.Tanh()
                                          )
        self.T_Sequential = nn.Sequential(nn.Linear(input_dim_T, hidden_dim),
                                          nn.Tanh()
                                          )
        self.C_Sequential = nn.Linear(hidden_dim, output_dim)

        self.CT_Sequential = nn.Linear(hidden_dim, output_dim)

    def init_weights(self):
        self.C_Sequential.weight = nn.Parameter(self.embedding_weight)

    def forward(self, img, text):
        y_I = self.I_Sequential(img)
        y_I_C = self.C_Sequential(y_I)
        y_T = self.T_Sequential(text)
        y_T_C = self.C_Sequential(y_T)
        return y_I, y_T, y_I_C, y_T_C


