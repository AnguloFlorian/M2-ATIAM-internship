# -*- coding: utf-8 -*-

import random
from tqdm import tqdm
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import CQTsDataset
from utils import triplet_loss, HiddenPrints
from model import ConvNet
from online_triplet_loss.losses import *

print('libraries imported')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = "/tsi/clusterhome/atiam-1005/music-structure-estimation/simple_supervised/"
data_path_harmonix = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts/*"
data_path_harmonix2 = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts_to_check/*"
data_path_personal = "/tsi/clusterhome/atiam-1005/data/Personal/cqts/*"
data_path_isoph = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts/*"

name_exp = "hard_mining_bis"
writer = SummaryWriter('{0}runs/{1}'.format(root_path, name_exp))

N_EPOCH = 250
batch_size = 6
files_cqts = glob.glob(data_path_isoph)
random.shuffle(files_cqts)

files_train = files_cqts[:int(0.8*len(files_cqts))]
files_val = files_cqts[int(0.8*len(files_cqts)):]

val_dataset = CQTsDataset(files_val)
validation_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
)



print(len(files_train), 'training examples')
print(len(files_val), 'validation examples')

model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters())
model.load_state_dict(torch.load('{0}weights/pretrained_model.pt'.format(root_path)))
best_loss = float('inf')

for epoch in range(N_EPOCH):
    running_loss = 0.0
    random.shuffle(files_train)
    train_dataset = CQTsDataset(files_train)
    train_loader = DataLoader(
          dataset=train_dataset,
          batch_size=batch_size,
          num_workers = 6
          )

    model.train()
    print("EPOCH " + str(epoch + 1) + "/" + str(N_EPOCH))
    for i, (cqts, labels) in enumerate(tqdm(train_loader)):
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        train_loss = 0
        for k in range(cqts.shape[0]):
            embeddings = model(cqts[k].to(device))
            with HiddenPrints():
                train_loss, _ = batch_all_triplet_loss(labels[k].to('cpu'), embeddings.to('cpu'), margin=0.1)
            running_loss += train_loss.item()
            train_loss = train_loss.to(device)
            train_loss.backward()
            running_loss += train_loss.item()
        optimizer.step()
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
        for i, (cqts, labels) in enumerate(tqdm(validation_loader)):
            train_loss = 0
            for k in range(cqts.shape[0]):
                embeddings = model(cqts[k].to(device))
                with HiddenPrints():
                    val_loss, _ = batch_all_triplet_loss(labels[k].to('cpu'), embeddings.to('cpu'), margin=0.1)
                running_loss += val_loss.item()
        # print statistics
        print('average validation loss: %.6f' %
              (running_loss / len(validation_loader)))
        # ...log the running loss
        writer.add_scalar('validation loss',
                          running_loss / len(validation_loader),
                          epoch)

# Save best model if validation loss is improved and update regularly last model state
    if running_loss <= best_loss:
        torch.save(model.state_dict(), "{0}weights/{1}_best.pt".format(root_path, name_exp))
        best_loss = running_loss
        
    torch.save(model.state_dict(), "{0}weights/{1}_last.pt".format(root_path, name_exp))
        
print('Finished Training')
