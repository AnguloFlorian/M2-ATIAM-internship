import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_same_pad import get_pad

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        kernel_conv = (4, 6)
        self.pad1 = get_pad((72, 512), kernel_conv)
        self.conv1 = nn.Conv2d(1, 64, kernel_conv)
        self.bnc1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4))
        self.pad2 = get_pad((36,128), kernel_conv)
        self.conv2 = nn.Conv2d(64, 128, kernel_conv)
        self.bnc2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 4))
        self.pad3 = get_pad((12,32), kernel_conv)
        self.conv3 = nn.Conv2d(128, 256, kernel_conv)
        self.bnc3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 4))
        self.fc1 = nn.Linear(6 * 8 * 256, 128)
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
        x = x.view(-1, 6 * 8 * 256)
        x = self.bnf1(F.relu(self.fc1(x)))
        x = self.bnf2(F.relu(self.fc2(x)))
        x = F.normalize(F.relu(self.fc3(x)), p=2)
        return x

    def forward(self, x):
        a = self.apply_cnn(x[:, 0, :, :])
        p = self.apply_cnn(x[:, 1, :, :])
        n = self.apply_cnn(x[:, 2, :, :])
        
        return a, p, n
    
    def inference(self, x):
        return self.apply_cnn(x)


class SSMnet(ConvNet):
    def __init__(self, batch_size=10, ssm_op='dot'):
        super(SSMnet, self).__init__()
        self.ssm_op = ssm_op
        
    def compute_ssm(self, embed_i, embed_j, ssm_op):
        if ssm_op == 'dot':
            return torch.dot(embed_i, embed_j)
        if ssm_op = 'sum':
            return torch.linalg.norm(embed_i - embed_j)

    def forward(self, x):
        # shape x : (n_beats, n_freq, n_time)
        n_beats = x.shape[0]
        embeds = torch.zeros(x.shape[0], 128)
        # Compute embeddings from all beats
        for i in range(0, x.shape[0], batch_size):
            embeds[i:(i + batch_size)] = super(SSMnet, self).forward(x[i:(i + batch_size)])
        
        ssm = torch.zeros()
        # Compute the SSM
        for i in range(n_beats):
            for j in range(n_beats):
                ssm[i, j] = compute_ssm(embeds[i], embeds[j], self.ssm_op)
        
        return ssm
