import torch
# from model import TextDNN, ImageDNN, DNN, CNN
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import numpy as np
from solver import Solver

np_seed = 1111
torch_seed = 2345

# torch.manual_seed(torch_seed)
np.random.seed(np_seed)

# torch.manual_seed(1111)  ### for my_extracted_wiki_feature
# np.random.seed(1111)  ### for my_extracted_wiki_feature


from custom_dataset import MyCustomDataset
from torch.utils.data import DataLoader
loss_type = 'mcml'

# test_size = 693
# idx = np.random.permutation(np.arange(test_size))
# idx = np.save('/root/workspace/datasets/wiki_data/extracted_features/my_indices.npy',idx)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch SS-C3MLNets Implementation')

#============ device ======================
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--device', type=int, default=0,
                    help='which device to use')

#============ model ======================
parser.add_argument('--input_dim_I', type=int, default=4096, metavar='N',
                    help='dim of image input')
parser.add_argument('--input_dim_T', type=int, default=300, metavar='N',
                    help='dim of text input')
parser.add_argument('--hidden_dim', type=int, default=1024, metavar='N',
                    help='dim of hidden layer')
parser.add_argument('--output_dim', type=int, default=1024, metavar='N',
                    help='dim of output layer')

#============ training ======================
parser.add_argument('--dataset_name', type=str, default='wiki_doc2vec_few_labels_split', metavar='N',
                    help='name of dataset')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--max_epoch', type=int, default=50, metavar='N',
                    help='how many epochs')
parser.add_argument('--test_epoch', type=int, default=1, metavar='S',
                    help='which epoch to print test result')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N',
                    help='which optimizer')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--loss_type', type=str, default='mcml', metavar='N',
                    help='which supervised loss')

parser.add_argument('--distribution_symmetry', type=bool, default=1,
                    help='the symmetry of the distributions')
parser.add_argument('--mask_out_digonal', type=bool, default=0,
                    help='whether mask out the digonal elements')

#============ parameter ======================
parser.add_argument('--alpha', type=float, default=1.0,
                    help='the balancing parameter of the supervised loss')
parser.add_argument('--beta', type=float, default=0.1,
                    help='the balancing parameter of the pairwise loss')
parser.add_argument('--gamma', type=int, default=1,
                    help='the proportion of the unlabeled data')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)


solver = Solver(args)
model = solver.train(args)