import torch
from model import TextDNN, ImageDNN, DNN
import numpy as np
from data import read_wiki
from custom_dataset import MyCustomDataset
from torch.utils.data import DataLoader

torch.manual_seed(1233)
np.random.seed(1233)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset = {x: MyCustomDataset(state=x)
           for x in ['train', 'test']}
dataloaders = {x: DataLoader(dataset[x], batch_size=100,
                             shuffle=True, num_workers=5)
               for x in ['train', 'test']}

dataset_sizes = {x: len(dataset[x]) for x in ['train', 'test']}



model = DNN(
    input_dim_I=128,
    input_dim_T=10,
    hidden_dim=50,
    output_dim=20)
model.to(device)

import train
model = train.train2(model, dataloaders, device, dataset_sizes, num_epochs=50)
