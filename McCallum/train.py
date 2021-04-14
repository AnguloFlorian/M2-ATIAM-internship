"/ids-cluster-storage/storage/atiam-1005/env_angulo/bin/python3"

import sys
import random
from copy import copy
import numpy as np
import librosa
import random
import warnings
import jams
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_same_pad import get_pad
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from dataloader import CQTsDataset
from utils import triplet_loss
from model import ConvNet

print('libraries imported')

device = torch.device("cuda:0" if torch.cuda.is_available() else "error")
root_path = "./"
data_path_train = "./cqts_harmonix/"
data_path_val = "./cqts_isoph/"

writer = SummaryWriter(root_path + "runs/experiment_train_harmonix")


N_EPOCH = 200
batch_size = 6
n_triplets = 16
n_files_train = glob.glob(data_path_train + "*")
n_files_val = glob.glob(data_path_val + "*")
#random.shuffle(n_files)
#split_idx = int(len(n_files)*0.8)
train_dataset = CQTsDataset(n_files_train, n_triplets)
val_dataset = CQTsDataset(n_files_val, n_triplets)

print(len(train_dataset), 'number of training examples')
print(len(val_dataset), 'number of validation examples')

model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters())

for epoch in range(N_EPOCH):
    running_loss = 0.0
    train_loader = DataLoader(
          dataset=train_dataset,
          batch_size=batch_size,
          num_workers = 6
          )
    model.train()
    print("EPOCH " + str(epoch + 1) + "/" + str(N_EPOCH))
    for i, data in enumerate(train_loader):
        #data = data[0].to(device)
        data = data.view(-1, 3, 72, 512).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        a, p, n = model(data)
        train_loss = triplet_loss(a, p, n, device)
        train_loss.backward()
        optimizer.step()
        running_loss += train_loss.item()
    # print statistics
    print('average train loss: %.6f' %
                (running_loss/len(train_loader)))
    # ...log the running loss
    writer.add_scalar('training loss',
                    running_loss / len(train_loader),
                    epoch)



    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        validation_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers = 6
        )
        for i, data in enumerate(validation_loader):
            data = data.view(-1, 3, 72, 512).to(device)
            a, p, n = model(data)
            val_loss = triplet_loss(a, p, n, device)
            running_loss += val_loss.item()
        # print statistics
        print('average validation loss: %.6f' %
              (running_loss / len(validation_loader)))
        # ...log the running loss
        writer.add_scalar('validation loss',
                          running_loss / len(validation_loader),
                          epoch)
    if epoch % 10 == 9:
        torch.save(model.state_dict(), root_path + "weights_harmonix/model" + str(epoch) + ".pt")
print('Finished Training')
