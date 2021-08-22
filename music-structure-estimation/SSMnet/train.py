# -*- coding: utf-8 -*-

import random
from tqdm import tqdm
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import CQTsDataset
from model import SSMnet
from utils import weighted_bce_loss
from madgrad import madgrad_wd

print('libraries imported')

device = torch.device("cuda:0" if torch.cuda.is_available() else "error")
root_path = "/tsi/clusterhome/atiam-1005/M2-ATIAM-internship/music-structure-estimation/SSMnet/"
data_path_harmonix = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts/*"
data_path_isoph = "/tsi/clusterhome/atiam-1005/data/Isophonics/cqts/*"

name_exp = "no_freeze_fc1_2fc(enfait3)_nm16(real_freeze)_pdist"
writer = SummaryWriter('{0}runs/{1}'.format(root_path, name_exp))


N_EPOCH = 250
batch_size = 1
backward_size = 6

files_train = glob.glob(data_path_harmonix)
files_val = glob.glob(data_path_isoph)

val_dataset = CQTsDataset(files_val)
validation_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
)

print(len(files_train), 'training examples')
print(len(files_val), 'validation examples')

model = SSMnet().to(device)

optimizer = madgrad_wd(model.parameters(), lr=5e-4, weight_decay=0.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

best_loss = float('inf')


print('evaluation untrained model')
with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_loss_cqts = 0.0
        for i, (cqts, ssm) in enumerate(tqdm(validation_loader)):
            embeds = model.apply_cnn(cqts.transpose(0, 1)).unsqueeze(0)
            cqts_flat = torch.nn.functional.normalize(cqts.reshape(1, -1, 72*64), p=2)
            _, T, _ = cqts_flat.shape
            #ssm_cqts = torch.zeros(T,T).cuda()
            #for j in range(T):
            #    for k in range(T):
            #        ssm_cqts[j, k] = torch.linalg.norm(cqts_flat[0, j]-cqts_flat[0,k], 2)
            ssm_cqts = torch.cdist(cqts_flat, cqts_flat)
            ssm_cqts = 1 - ssm_cqts/torch.max(ssm_cqts)
            #ssm_hat = 0.5*(1 + torch.bmm(embeds, embeds.transpose(-2, -1)))
            ssm_hat = torch.cdist(embeds, embeds)
            ssm_hat = 1 - ssm_hat/torch.max(ssm_hat)
            val_loss = weighted_bce_loss(ssm_hat, ssm)
            cqts_loss = weighted_bce_loss(ssm_cqts, ssm)
            running_loss += val_loss.item()
            running_loss_cqts += cqts_loss.item()
        # print statistics
        print('average validation loss (Isophonics): %.6f' %
              (running_loss / len(validation_loader)))
        print('average cqts loss (Isophonics): %.6f' %
              (running_loss_cqts / len(validation_loader)))

model.load_state_dict(torch.load('{0}weights/best_a0.2nm16.pt'.format(root_path)), strict=False)

print('evaluating model with pretrained embeddings')

with torch.no_grad():
        model.eval()
        running_loss = 0.0
        for i, (cqts, ssm) in enumerate(tqdm(validation_loader)):
            embeds = model.apply_cnn(cqts.transpose(0, 1)).unsqueeze(0)
            cqts_flat = torch.nn.functional.normalize(cqts.reshape(1, -1, 72*64), p=2)
            _, T, _ = cqts_flat.shape
            #ssm_cqts = torch.zeros(T,T).cuda()
            #for j in range(T):
            #    for k in range(T):
            #        ssm_cqts[j, k] = torch.linalg.norm(cqts_flat[0, j]-cqts_flat[0,k], 2)
            ssm_cqts = torch.cdist(cqts_flat, cqts_flat)
            ssm_cqts = 1 - ssm_cqts/torch.max(ssm_cqts)
            #ssm_hat = 0.5*(1 + torch.bmm(embeds, embeds.transpose(-2, -1)))
            ssm_hat = torch.cdist(embeds, embeds)
            ssm_hat = 1 - ssm_hat/torch.max(ssm_hat)
            val_loss = weighted_bce_loss(ssm_hat, ssm)
            cqts_loss = weighted_bce_loss(ssm_cqts, ssm)
            running_loss += val_loss.item()
            running_loss_cqts += cqts_loss.item()
        # print statistics
        print('average validation loss (Isophonics): %.6f' %
              (running_loss / len(validation_loader)))



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
        ssm_hat = model(cqts)
        train_loss = weighted_bce_loss(ssm_hat, ssm)
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
            ssm_hat = model(cqts)
            val_loss = weighted_bce_loss(ssm_hat, ssm)
            running_loss += val_loss.item()
        # print statistics
        print('average validation loss (Isophonics): %.6f' %
              (running_loss / len(validation_loader)))
        # ...log the running loss
        writer.add_scalar('validation loss (Isophonics)',
                          running_loss / len(validation_loader),
                          epoch)

    scheduler.step(running_loss)
    # Save best model if validation loss is improved and update regularly last model state
    if running_loss <= best_loss:
        torch.save(model.state_dict(), "{0}weights/{1}_best.pt".format(root_path, name_exp))
        best_loss = running_loss
    
    torch.save(model.state_dict(), "{0}weights/{1}_last.pt".format(root_path, name_exp))
        
print('Finished Training')
