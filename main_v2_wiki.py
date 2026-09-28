import torch
# from model import TextDNN, ImageDNN, DNN, CNN
from models.model_CNN_wiki import CNN
import numpy as np

np_seed = 1233 # 1111
torch_seed = 1233 # 2345

torch.manual_seed(torch_seed)
np.random.seed(np_seed)

# torch.manual_seed(1111)  ### for my_extracted_wiki_feature
# np.random.seed(1111)  ### for my_extracted_wiki_feature


from custom_dataset import MyCustomDataset
from torch.utils.data import DataLoader

loss_type = 'mcml'

# test_size = 693
# idx = np.random.permutation(np.arange(test_size))
# idx = np.save('/root/workspace/datasets/wiki_data/extracted_features/my_indices.npy',idx)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_config = {
    'dataset_name': 'wiki_deep',
    'class_number': 200}

dataset = {x: MyCustomDataset(dataset=dataset_config['dataset_name'], state=x)
           for x in ['train', 'test']}
dataloaders = {x: DataLoader(dataset[x], batch_size=50,
                             shuffle=True, num_workers=10)
               for x in ['train', 'test']}

dataset_sizes = {x: len(dataset[x]) for x in ['train', 'test']}


model = CNN(
    input_dim_I=4096,
    input_dim_T=300,
    hidden_dim=1024,
    output_dim=1024
)
model.to(device)

import train_wiki as train
model = train.train2(model, dataloaders, device, dataset_sizes, loss_type, num_epochs=200)