import torch
# from model import TextDNN, ImageDNN, DNN, CNN
from model_pretrained_wordcnn import TextCNN
import numpy as np
torch.manual_seed(1233)
np.random.seed(1233)

from custom_dataset import MyCustomDataset
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_name='nus_deep'

dataset = {x: MyCustomDataset(dataset=dataset_name, state=x)
           for x in ['train', 'test']}
dataloaders = {x: DataLoader(dataset[x], batch_size=50,
                             shuffle=True, num_workers=10)
               for x in ['train', 'test']}

dataset_sizes = {x: len(dataset[x]) for x in ['train', 'test']}


wv_matrix = torch.FloatTensor(dataset['train'].wv_matrix, device=device)
model = TextCNN(
    mode='non-multichannel',
    in_channels=1,
    kernel_number=100,
    voca_size=wv_matrix.shape[0],
    embed_dim=wv_matrix.shape[1],
    kernel_sizes=[3, 4, 5],
    dropout_p=0.6,
    embedding_weight=wv_matrix,
    outpt_dim=1024,
    class_number=10)
model.to(device)

import train_pretrained_wordcnn
model = train_pretrained_wordcnn.train2(model, dataloaders, device, dataset_sizes, num_epochs=40)

# torch.save(model.state_dict(), 'pretrained_wordcnn/'+dataset_name+'_pretrained_wordcnn.tar')

torch.save(model, 'pretrained_wordcnn/'+dataset_name+'_pretrained_wordcnn.pt')