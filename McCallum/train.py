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

print('libraries imported')

device = torch.device("cuda:0" if torch.cuda.is_available() else "error")
root_path = "/tsi/clusterhome/atiam-1005/music-structure-estimation/McCallum/"
data_path_harmonix = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts/*"
data_path_harmonix2 = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts_to_check/*"
data_path_personal = "/tsi/clusterhome/atiam-1005/data/Personal/cqts/*"
data_path_jamendo = "/tsi/clusterhome/atiam-1005/data/Jamendo/cqts/*"
data_path_isoph = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts/*"


name_exp = "norm_quite_small_alternative"

writer = SummaryWriter(root_path + "runs/" + name_exp)


N_EPOCH = 250
batch_size = 6
n_batchs = 256
n_triplets = 16
dim_cqts = (72, 64)
n_files_train = glob.glob(data_path_jamendo)
#n_files_train.extend(glob.glob(data_path_harmonix))
#n_files_train.extend(glob.glob(data_path_harmonix2))
n_files_train.extend(glob.glob(data_path_personal))
n_files_val = glob.glob(data_path_isoph)
n_files_val.extend(glob.glob(data_path_harmonix2))
n_files_val.extend(glob.glob(data_path_harmonix))

val_dataset = CQTsDataset(n_files_val, n_triplets=n_triplets)


print(len(n_files_train), 'training examples')
print(len(n_files_val), 'validation examples')

model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters())
best_loss = float('inf')
torch.save(model.state_dict(), "{0}weights/{1}_init.pt".format(root_path, name_exp))
for epoch in range(N_EPOCH):
    running_loss = 0.0
    random.shuffle(n_files_train)
    train_dataset = CQTsDataset(n_files_train[:batch_size*n_batchs], n_triplets=n_triplets)
    train_loader = DataLoader(
          dataset=train_dataset,
          batch_size=batch_size,
          num_workers = 6,
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
    
    # Save best model if validation loss is improved and update regularly last model state
    if val_loss <= best_loss:
        torch.save(model.state_dict(), "{0}weights/{1}_best.pt".format(root_path, name_exp))
        best_loss = val_loss
    if epoch % 10 == 9:
        torch.save(model.state_dict(), "{0}weights/{1}_last.pt".format(root_path, name_exp))
print('Finished Training')
