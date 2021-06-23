import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_same_pad import get_pad

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        kernel_conv = (6, 4)
        self.pad1 = get_pad((72, 64), kernel_conv)
        self.conv1 = nn.Conv2d(1, 16, kernel_conv)
        self.bnc1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pad2 = get_pad((36, 32), kernel_conv)
        self.conv2 = nn.Conv2d(16, 32, kernel_conv)
        self.bnc2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pad3 = get_pad((18,16), kernel_conv)
        self.conv3 = nn.Conv2d(32, 64, kernel_conv)
        self.bnc3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 4))
        self.fc1 = nn.Linear(6 * 4 * 64, 512)
        self.bnf1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bnf2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bnf3 = nn.BatchNorm1d(128)
        
    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.pool1(self.bnc1(F.relu(self.conv1(F.pad(x,self.pad1)))))
        x = self.pool2(self.bnc2(F.relu(self.conv2(F.pad(x,self.pad2)))))
        x = self.pool3(self.bnc3(F.relu(self.conv3(F.pad(x,self.pad3)))))
        x = x.view(-1, 6 * 4 * 64)
        x = self.bnf1(F.relu(self.fc1(x)))
        x = self.bnf2(F.relu(self.fc2(x)))
        x = F.normalize(self.bnf3(F.relu(self.fc3(x))), p=2)
        
        return x