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
from model import SSMnet

print('libraries imported')

device = torch.device("cuda:0" if torch.cuda.is_available() else "error")
root_path = "/tsi/clusterhome/atiam-1005/music-structure-estimation/SSMnet/"
data_path_harmonix = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts/*"
data_path_harmonix2 = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts_to_check/*"
data_path_personal = "/tsi/clusterhome/atiam-1005/data/Personal/cqts/*"
data_path_isoph = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts/*"

name_exp = "bce_loss_sum_freezed"
writer = SummaryWriter('{0}runs/{1}'.format(root_path, name_exp))


N_EPOCH = 250
batch_size = 1
backward_size = 6
n_batchs = 256
n_triplets = 16
dim_cqts = (72, 64)
files_train = glob.glob(data_path_isoph)
random.shuffle(files_train)
files_val = files_train[int(0.8*len(files_train)):]
files_train = files_train[:int(0.8*len(files_train))]


val_dataset = CQTsDataset(files_val)
validation_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
)

print(len(files_train), 'training examples')
print(len(files_val), 'validation examples')

model = SSMnet().to(device)
optimizer = optim.Adam(model.parameters())
mse_loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCELoss()
model.load_state_dict(torch.load('{0}weights/pretrained_model.pt'.format(root_path)))
best_loss = float('inf')

for epoch in range(N_EPOCH):
    running_loss = 0.0
    random.shuffle(files_train)
    train_dataset = CQTsDataset(files_train)
    train_loader = DataLoader(
          dataset=train_dataset,
          batch_size=batch_size,
          )

    model.train()
    print("EPOCH " + str(epoch + 1) + "/" + str(N_EPOCH))
    for i, (cqts, ssm) in enumerate(tqdm(train_loader)):
        ssm_hat = model(cqts.squeeze())
        train_loss = bce_loss(ssm_hat, ssm.squeeze())
        train_loss.backward()
        running_loss += train_loss.item()
        if (i + 1) % backward_size == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    optimizer.step()
    optimizer.zero_grad()
    
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
        for i, (cqts, ssm) in enumerate(tqdm(validation_loader)):
            ssm_hat = model(cqts.squeeze().to(device))
            val_loss = bce_loss(ssm_hat, ssm.squeeze())
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
