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
from util.util import visualise_geometries, save_vis

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

visualise_geometries(dataset[0][0], dataset[0][3])

# # Calculate the total number of batches
# num_batches = len(dataloader)

# # Calculate the total number of data samples
# total_samples = num_batches * opt.batchSize

# print("Number of batches in the dataloader:", num_batches)
# print("Total number of data samples in the dataloader:", total_samples)

# # Iterate over the dataloader to get the first batch
# for i, batch in enumerate(dataloader):
#     if i == 0:
#         sample = batch
#         break

# # Printing the number of samples in the batch
# print("Number of samples in the batch:", len(sample))

# # Displaying the content of each sample in the batch
# for i, data in enumerate(sample):
#     visualise_geometries(data)
#     print(f"Sample {i + 1}:")
#     if isinstance(data, torch.Tensor):
#         print("Tensor shape:", data.shape)
#         print(data)
#     else:
#         print("Data type:", type(data))
#         print(data)
#     print()