import numpy as np
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
        x = self.pool1(self.bnc1(F.relu(self.conv1(F.pad(x,self.pad1)))))
        x = self.pool2(self.bnc2(F.relu(self.conv2(F.pad(x,self.pad2)))))
        x = self.pool3(self.bnc3(F.relu(self.conv3(F.pad(x,self.pad3)))))
        x = x.view(-1, self.resize)
        #x = self.bnf1(F.relu(self.fc1(x)))
        #x = self.bnf2(F.relu(self.fc2(x)))
        #x = F.normalize(self.bnf3(F.relu(self.fc3(x))), p=2)
        x = F.normalize(self.bnf1(F.relu(self.fc1(x))), p=2)
        
        return x

    
    def inference(self, x):
        return self.apply_cnn(x)


class SSMnet(ConvNet):
    def __init__(self, freeze_convs=True, embed_dim=32, n_classes=8):
        super(SSMnet, self).__init__()
        self.sqrt_dim = np.sqrt(embed_dim)
        
        
        
        # attention layers
        self.fc_query = nn.Linear(128, embed_dim)
        self.fc_query1 = nn.Linear(128, embed_dim * 2)
        self.fc_query2 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc_key = nn.Linear(128, embed_dim)
        self.fc_key1 = nn.Linear(128, embed_dim * 2)
        self.fc_key2 = nn.Linear(embed_dim * 2, embed_dim)
        

        self.fc_pred1 = nn.Linear(128, embed_dim)
        self.fc_pred2 = nn.Linear(embed_dim, n_classes)
        
        
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
            #self.freeze_layer(self.fc2)
            #self.freeze_layer(self.bnf2)
            #self.freeze_layer(self.fc3)
            #self.freeze_layer(self.bnf3)
        
    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False
    
    
    def forward(self, x, infer=False):
        # Compute embeddings from all beats
        embeds = super(SSMnet, self).apply_cnn(x.transpose(0, 1)).unsqueeze(0)
        # Compute value and attention vectors
        #query =  torch.tanh(self.bn_query(self.fc_query(embeds)))
        #key =  torch.tanh(self.bn_key(self.fc_key(embeds)))
        #value =  torch.tanh(self.bn_value(self.fc_value(embeds)))
        query = F.relu(self.fc_query1(embeds))
        query = torch.tanh(self.fc_query2(query))
        key =  F.relu(self.fc_key1(embeds))
        key =  torch.tanh(self.fc_key2(key))
        
        # Compute attention scores
        score = torch.bmm(query, key.transpose(-1, -2)) / self.sqrt_dim
        attention_scores = F.softmax(score, dim=-1)
        context = torch.bmm(attention_scores, embeds)
        
        prob_labels = F.relu(self.fc_pred1(context))
        prob_labels = F.softmax(self.fc_pred2(prob_labels), dim=-1)
        
        if infer:
            return prob_labels, attention_scores
            
        else:   
            ssm = torch.bmm(prob_labels, prob_labels.transpose(-2, -1))
            return ssm, prob_labels, attention_scores
