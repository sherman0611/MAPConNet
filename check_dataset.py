import os
import sys
import pymesh
import torch

from data.human_data import SMPL_DATA
from data.animal_data import SMAL_DATA
from ver2ver_trainer import Ver2VerTrainer
from options.train_options import TrainOptions
from util.iter_counter import IterationCounter
from util.util import print_current_errors


# parse options
opt = TrainOptions().parse()
# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
if opt.dataset_mode == 'human':
    dataset = SMPL_DATA(opt, True)
    if opt.use_unlabelled:
        dataset_unlabelled = SMPL_DATA(opt, True, False)
elif opt.dataset_mode == 'animal':
    dataset = SMAL_DATA(opt, True)
    if opt.use_unlabelled:
        dataset_unlabelled = SMAL_DATA(opt, True, False)
else:
    raise ValueError("|dataset_mode| is invalid")

if opt.percentage > 0:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.nThreads), drop_last=opt.isTrain)
    iter_data = iter(dataloader)
if opt.use_unlabelled:
    dataloader_unlabelled = torch.utils.data.DataLoader(dataset_unlabelled, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.nThreads), drop_last=opt.isTrain)
    iter_data_unlabelled = iter(dataloader_unlabelled)

data = next(iter_data)
print(data)
print(data.shape)