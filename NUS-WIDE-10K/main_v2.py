import torch
# from model import TextDNN, ImageDNN, DNN, CNN
from model_CNN import CNN
import numpy as np
torch.manual_seed(1233)
np.random.seed(1233)

from custom_dataset import MyCustomDataset
from torch.utils.data import DataLoader


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset_config = {
    'dataset_name': 'nus_deep',
    'class_number': 10,
    'pretrained_wordcnn_path': 'pretrained_wordcnn/nus_deep_pretrained_wordcnn.pt'}

dataset = {x: MyCustomDataset(dataset=dataset_config['dataset_name'], state=x)
           for x in ['train', 'test']}
dataloaders = {x: DataLoader(dataset[x], batch_size=50,
                             shuffle=True, num_workers=10)
               for x in ['train', 'test']}

dataset_sizes = {x: len(dataset[x]) for x in ['train', 'test']}


wv_matrix = torch.FloatTensor(dataset['train'].wv_matrix, device=device)
model = CNN(
    input_dim_I=4096,
    embedding_weight=wv_matrix,
    hidden_dim1=1024,
    hidden_dim2=1024,
    output_dim=1024,
    class_number=dataset_config['class_number'],
    # pretrained_wordcnn_path=dataset_config['pretrained_wordcnn_path']
    pretrained_wordcnn_path=None
)
model.to(device)

import train
model = train.train2(model, dataloaders, device, dataset_sizes, num_epochs=200)
