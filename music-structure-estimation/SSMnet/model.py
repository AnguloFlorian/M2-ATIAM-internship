import torch
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
        
    def apply_cnn(self,x):
        x = x.unsqueeze(1) # for 1 channel
        x = self.pool1(self.bnc1(F.relu(self.conv1(F.pad(x,self.pad1)))))
        x = self.pool2(self.bnc2(F.relu(self.conv2(F.pad(x,self.pad2)))))
        x = self.pool3(self.bnc3(F.relu(self.conv3(F.pad(x,self.pad3)))))
        x = x.view(-1, 6 * 4 * 64)
        x = self.bnf1(F.relu(self.fc1(x)))
        x = self.bnf2(F.relu(self.fc2(x)))
        x = F.normalize(self.bnf3(F.relu(self.fc3(x))), p=2)
        
        return x

    def forward(self, x):
        a = self.apply_cnn(x[:, 0, :, :])
        p = self.apply_cnn(x[:, 1, :, :])
        n = self.apply_cnn(x[:, 2, :, :])
        
        return a, p, n
    
    def inference(self, x):
        return self.apply_cnn(x)


class SSMnet(ConvNet):
    def __init__(self, freeze_convs=True, ssm_op='sum'):
        super(SSMnet, self).__init__()
        self.ssm_op = ssm_op
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
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
            self.freeze_layer(self.fc1)
            self.freeze_layer(self.bnf1)
            self.freeze_layer(self.fc2)
            self.freeze_layer(self.bnf2)
            
            
    def compute_ssm(self, embed_i, embed_j):
        if self.ssm_op == 'dot':
            return torch.dot(embed_i, embed_j)
        if self.ssm_op == 'sum':
            return torch.linalg.norm(embed_i - embed_j)
    
    
    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False
    
    
    def forward(self, x):
        # shape x : (n_beats, n_freq, n_time)
        n_beats = x.shape[0]
        embeds = torch.zeros(x.shape[0], 128).to(self.device)
        # Compute embeddings from all beats
        embeds = super(SSMnet, self).apply_cnn(x)
        ssm = torch.zeros((n_beats, n_beats)).to(self.device)
        # Compute the SSM
        for i in range(n_beats):
            for j in range(n_beats):
                ssm[i, j] = self.compute_ssm(embeds[i], embeds[j])
        
        ssm = ssm/torch.max(ssm)
        
        return ssm
