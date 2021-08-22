# -*- coding: utf-8 -*-

import random
from tqdm import tqdm
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import CQTsDataset
from utils import triplet_loss
from model import ConvNet
from madgrad import madgrad_wd

print('libraries imported')

device = torch.device("cuda:0" if torch.cuda.is_available() else "error")
root_path = "/tsi/clusterhome/atiam-1005/M2-ATIAM-internship/music-structure-estimation/McCallum/"
data_path_harmonix = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts/*"
data_path_personal = "/tsi/clusterhome/atiam-1005/data/Personal/cqts/*"
data_path_isoph = "/tsi/clusterhome/atiam-1005/data/Isophonics/cqts/*"

name_exp = "less_fc_a0.25_lr1e-4_wd_1e-2_group_nmin16"
writer = SummaryWriter('{0}runs/{1}'.format(root_path, name_exp))


N_EPOCH = 250
batch_size = 6
n_batches = 256
n_triplets = 16

files_train = glob.glob(data_path_personal)
files_val = glob.glob(data_path_isoph)
files_val.extend(glob.glob(data_path_harmonix))

val_dataset = CQTsDataset(files_val)
validation_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
)

print(len(files_train), 'training examples')
print(len(files_val), 'validation examples')
model = ConvNet().to(device)
#optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = madgrad_wd(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
best_loss = float('inf')

for epoch in range(N_EPOCH):
    running_loss = 0.0
    random.shuffle(files_train)
    print(files_train[0], files_train[n_batches*batch_size])
    train_dataset = CQTsDataset(files_train[:n_batches*batch_size])
    train_loader = DataLoader(
          dataset=train_dataset,
          batch_size=batch_size,
          num_workers = 6
          )

    model.train()
    print("EPOCH " + str(epoch + 1) + "/" + str(N_EPOCH))
    for i, data in enumerate(tqdm(train_loader)):
        data = data.view(-1, 3, 72, 64).to(device)
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
        for i, data in enumerate(tqdm(validation_loader)):
            data = data.view(-1, 3, 72, 64).to(device)
            a, p, n = model(data)
            val_loss = triplet_loss(a, p, n, device)
            running_loss += val_loss.item()
        # print statistics
        print('average validation loss (Isophonics): %.6f' %
              (running_loss / len(validation_loader)))
        # ...log the running loss
        writer.add_scalar('validation loss (Isophonics)',
                          running_loss / len(validation_loader),
                          epoch)

    #scheduler.step(running_loss)
    # Save best model if validation loss is improved and update regularly last model state
    if running_loss <= best_loss:
        torch.save(model.state_dict(), "{0}weights/{1}_best.pt".format(root_path, name_exp))
        best_loss = running_loss
    
    torch.save(model.state_dict(), "{0}weights/{1}_last.pt".format(root_path, name_exp))
        
print('Finished Training')
