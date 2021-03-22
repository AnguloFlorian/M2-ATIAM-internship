import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from copy import copy
import librosa
import random
import warnings
import jams
import glob



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = '/ldaphome/atiam-1005/music-structure-estimation/data/Isophonics/'


writer = SummaryWriter('runs/')



N_EPOCH = 100
batch_size = 6
n_triplets = 16
dataset = CQTsDataset(root_path, n_triplets)
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters())
model.load_state_dict(torch.load(root_path + "model.pt"))

for epoch in range(48, N_EPOCH):
    running_loss = 0.0
    data_loader = DataLoader(
          dataset=dataset,
          batch_size=batch_size,
          )
    model.train()
    print("EPOCH " + str(epoch + 1) + "/" + str(N_EPOCH))
    for i, data in enumerate(data_loader):
        #data = data[0].to(device)
        data = data.view(-1, 3, 72, 512).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        a, p, n = model(data)
        loss = triplet_loss(a, p, n)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i + 1) % 10 == 0:    # print every 5 mini-batches
            print('[%d, %5d]  average loss: %.6f' %
                (epoch + 1, i + 1, running_loss/10))
            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 10,
                            epoch * len(data_loader) + i)

            
            running_loss = 0.0
    if epoch % 10 = 9:
        torch.save(model.state_dict(), root_path + "model" + str(epoch) + ".pt")
print('Finished Training')