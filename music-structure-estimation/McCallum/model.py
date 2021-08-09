import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_same_pad import get_pad

class ConvNet(nn.Module):
    def __init__(self, big_conv=False):
        super(ConvNet, self).__init__()
        kernel_conv1 = (12, 4) if big_conv else (6, 4)
        kernel_conv2 = (6, 4) if big_conv else (6, 4)
        kernel_conv3 = (3, 2) if big_conv else (6, 4)
        # kernels (12, 4) (6, 4) (3, 2)
        self.pad1 = get_pad((72, 64), kernel_conv1) 
        self.conv1 = nn.Conv2d(1, 32, kernel_conv1)
        self.bnc1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4))
        self.pad2 = get_pad((36, 16), kernel_conv2)
        self.conv2 = nn.Conv2d(32, 64, kernel_conv2)
        self.bnc2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 4))
        self.pad3 = get_pad((12, 4), kernel_conv3)
        self.conv3 = nn.Conv2d(64, 128, kernel_conv3)
        self.bnc3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 2))
        self.resize = 4 * 2 * 128
        self.fc1 = nn.Linear(self.resize, 128)
        self.bnf1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bnf2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.bnf3 = nn.BatchNorm1d(128)
        
    def apply_cnn(self,x):
        x = x.unsqueeze(1) # for 1 channel
        x = self.pool1(self.bnc1(F.relu(self.conv1(F.pad(x,self.pad1)))))
        x = self.pool2(self.bnc2(F.relu(self.conv2(F.pad(x,self.pad2)))))
        x = self.pool3(self.bnc3(F.relu(self.conv3(F.pad(x,self.pad3)))))
        x = x.view(-1, self.resize)
        #x = self.bnf1(F.relu(self.fc1(x)))
        #x = self.bnf2(F.relu(self.fc2(x)))
        #x = F.normalize(self.bnf3(F.relu(self.fc3(x))), p=2)
        x = F.normalize(self.bnf1(F.relu(self.fc1(x))), p=2)
        
        return x

    def forward(self, x):
        a = self.apply_cnn(x[:, 0, :, :])
        p = self.apply_cnn(x[:, 1, :, :])
        n = self.apply_cnn(x[:, 2, :, :])
        
        return a, p, n
    
    def inference(self, x):
        return self.apply_cnn(x)