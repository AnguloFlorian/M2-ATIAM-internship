# -*- coding: utf-8 -*-

import random
from tqdm import tqdm
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import SSMsDataset
from model import CohenNet
from madgrad import madgrad_wd

print('libraries imported')

device = torch.device("cuda:0" if torch.cuda.is_available() else "error")
root_path = "/tsi/clusterhome/atiam-1005/M2-ATIAM-internship/music-structure-estimation/Cohen-Hadria/"
data_path_harmonix = "/tsi/clusterhome/atiam-1005/data/Harmonix/ssm/McCallum/*.npy"
data_path_isoph = "/tsi/clusterhome/atiam-1005/data/Isophonics/ssm/McCallum/*.npy"

name_exp = "final_exp_McCallum"
writer = SummaryWriter('{0}runs/{1}'.format(root_path, name_exp))


N_EPOCH = 50
batch_size = 4

files_train = glob.glob(data_path_harmonix)
files_val = glob.glob(data_path_isoph)

val_dataset = SSMsDataset(files_val)
validation_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
)

print(len(files_train), 'training examples')
print(len(files_val), 'validation examples')

model = CohenNet().to(device)

optimizer = madgrad_wd(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
bce_loss = torch.nn.BCELoss()
best_loss = float('inf')

with torch.no_grad():
    model.eval()
    running_loss = 0.0
    for i, (ssm, boundaries) in enumerate(tqdm(validation_loader)):
        ssm = ssm.view(-1, 15, 15)
        boundaries = boundaries.view(-1, 1)
        boundaries = boundaries*0.7 + (1-boundaries)*0.3
        probs = model(ssm)
        val_loss = bce_loss(probs, boundaries)
        running_loss += val_loss.item()
        optimizer.step()
        optimizer.zero_grad()
    # print statistics
    print('average validation loss (Isophonics): %.6f' %
          (running_loss / len(validation_loader)))


for epoch in range(N_EPOCH):
    running_loss = 0.0
    random.shuffle(files_train)
    train_dataset = SSMsDataset(files_train)
    train_loader = DataLoader(
          dataset=train_dataset,
          batch_size=batch_size,
          )
    model.train()
    print("EPOCH " + str(epoch + 1) + "/" + str(N_EPOCH))
    for i, (ssm, boundaries) in enumerate(tqdm(train_loader)):
        ssm = ssm.view(-1, 15, 15)
        boundaries = boundaries.view(-1, 1)
        boundaries = boundaries*0.7 + (1-boundaries)*0.3
        probs = model(ssm)
        train_loss = bce_loss(probs, boundaries)
        train_loss.backward()
        running_loss += train_loss.item()
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
        for i, (ssm, boundaries) in enumerate(tqdm(validation_loader)):
            ssm = ssm.view(-1, 15, 15)
            boundaries = boundaries.view(-1, 1)
            probs = model(ssm)
            boundaries = boundaries*0.7 + (1-boundaries)*0.3
            val_loss = bce_loss(probs, boundaries)
            running_loss += val_loss.item()
            optimizer.step()
            optimizer.zero_grad()
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
