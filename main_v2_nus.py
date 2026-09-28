import torch
# from model import TextDNN, ImageDNN, DNN, CNN
from model_CNN_mcsm import CNN
import numpy as np
np_seed = 1234
torch_seed = 1234

torch.manual_seed(torch_seed)
np.random.seed(np_seed)



from custom_dataset import MyCustomDataset
from torch.utils.data import DataLoader


# test_size = 693
# idx = np.random.permutation(np.arange(test_size))
# idx = np.save('/root/workspace/datasets/wiki_data/extracted_features/my_indices.npy',idx)



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset_config = {
    'dataset_name': 'nus_deep',
    'class_number': 10}

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

import train
model = train.train2(model, dataloaders, device, dataset_sizes, num_epochs=40)