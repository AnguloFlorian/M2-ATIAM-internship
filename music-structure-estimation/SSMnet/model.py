import torch
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
        self.bnc1 = nn.GroupNorm(32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4))
        self.pad2 = get_pad((36, 16), kernel_conv2)
        self.conv2 = nn.Conv2d(32, 64, kernel_conv2)
        self.bnc2 = nn.GroupNorm(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 4))
        self.pad3 = get_pad((12, 4), kernel_conv3)
        self.conv3 = nn.Conv2d(64, 128, kernel_conv3)
        self.bnc3 = nn.GroupNorm(32, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 2))
        self.resize = 4 * 2 * 128
        self.fc1 = nn.Linear(self.resize, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.selu = nn.SELU()
    def apply_cnn(self,x):
        #x = x.unsqueeze(1) # for 1 channel
        x = self.pool1(self.bnc1(self.selu(self.conv1(F.pad(x,self.pad1)))))
        x = self.pool2(self.bnc2(self.selu(self.conv2(F.pad(x,self.pad2)))))
        x = self.pool3(self.bnc3(self.selu(self.conv3(F.pad(x,self.pad3)))))
        x = x.view(-1, self.resize)
        #x = self.selu(self.fc1(x))
        #x = self.selu(self.fc2(x))
        #x = F.normalize(self.selu(self.fc3(x)), p=2)
        x = F.normalize(self.selu(self.fc1(x)), p=2)
        return x

    def forward(self, x):
        a = self.apply_cnn(x[:, 0, :, :])
        p = self.apply_cnn(x[:, 1, :, :])
        n = self.apply_cnn(x[:, 2, :, :])

        return a, p, n

    def inference(self, x):
        return self.apply_cnn(x)


class SSMnet(ConvNet):
    def __init__(self, freeze_convs=True):
        super(SSMnet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc2sm = nn.Linear(128, 96)
        self.fc3sm = nn.Linear(96, 64)
                
        if freeze_convs:
            self.freeze_layer(self.conv1)
            self.freeze_layer(self.bnc1)
            self.freeze_layer(self.pool1)
            self.freeze_layer(self.conv2)
            self.freeze_layer(self.bnc2)
            self.freeze_layer(self.pool2)
            self.freeze_layer(self.conv3)
            self.freeze_layer(self.bnc3)
            self.freeze_layer(self.pool3)
            #self.freeze_layer(self.fc1)
    
    
    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False
    
    
    def forward(self, x):
        # Compute embeddings from all beats
        embeds = super(SSMnet, self).apply_cnn(x.transpose(0, 1)).unsqueeze(0) 
        #embeds = F.normalize(self.selu(self.fc2sm(embeds)), p=2)
        #embeds = F.normalize(self.selu(self.fc3sm(embeds)), p=2)
        # Compute the SSM
        smm_hat = torch.cdist(embeds, embeds, 2)
        smm_hat = 1 - smm_hat/torch.max(smm_hat)
        return smm_hat
