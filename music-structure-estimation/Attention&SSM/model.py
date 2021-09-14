import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable
from torch_same_pad import get_pad

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=2500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

"""
class GlobalAttention(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        #self.mask.shape = (1, 1, max_len, max_len)

        self.pos = PositionalEncoding(d_model, dropout, max_len)
        mask = torch.tril(torch.ones(1, 1, max_len, max_len))
        self.register_buffer(
            "mask", 
            mask
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )
        
        k_t = self.key(x).transpose(-1,-2)
        # k_t.shape = (batch_size, d_model, seq_len)
        v = self.value(x)
        q = self.query(x)
        # shape = (batch_size, seq_len, d_model)
        
        Pos = self.pos(x).transpose(-1, -2)
        # Pos.shape = (batch_size, d_model, seq_len)        
        QK_t = torch.matmul(q, k_t)
        QPos = torch.matmul(q, Pos)
        # QK_t.shape = (batch_size, seq_len, seq_len)
        # Pos.shape = (batch_size, seq_len, seq_len)
        attn = (QK_t + QPos) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.dropout(out)


class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=2048, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        mask = torch.tril(torch.ones(1, 1, max_len, max_len))
        self.register_buffer(
            "mask", 
            mask
        )
        self.identity = nn.Identity()
        #self.mask.shape = (1, 1, max_len, max_len)
          
    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        residual = self.identity(x)
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )
        
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)
        
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out += residual
        return out
        
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel

"""

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
        self.selu = nn.SELU()
    
    def apply_cnn(self,x):
        #x = x.unsqueeze(1) # for 1 channel
        x = self.pool1(self.bnc1(self.selu(self.conv1(F.pad(x,self.pad1)))))
        x = self.pool2(self.bnc2(self.selu(self.conv2(F.pad(x,self.pad2)))))
        x = self.pool3(self.bnc3(self.selu(self.conv3(F.pad(x,self.pad3)))))
        x = x.view(-1, self.resize)
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
    def __init__(self, freeze_convs=False, d_model=128, num_heads=1, max_len=2048, dropout=0):
        super(SSMnet, self).__init__()
        self.sqrt_dim = np.sqrt(d_model)
        #self.rel_att = RelativeGlobalAttention(d_model, num_heads)
        #self.glob_att = GlobalAttention(d_model, max_len)
        self.max_len = max_len
        self.d_model = d_model
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.pos = PositionalEncoding(d_model, dropout, max_len)  
        self.identity = nn.Identity()
        #self.mh_att = nn.MultiheadAttention(embed_dim=d_model, num_heads=4)
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

    def freeze_layer(self, layer):
        for p in layer.parameters():
            p.requires_grad = False        
            
    def zero_grad_cnn(self):
        self.conv1.zero_grad()
        self.bnc1.zero_grad()
        self.pool1.zero_grad()
        self.conv2.zero_grad()
        self.bnc2.zero_grad()
        self.pool2.zero_grad()
        self.conv3.zero_grad()
        self.bnc3.zero_grad()
        self.pool3.zero_grad()
        self.fc1.zero_grad()
    
    
    def forward(self, x, infer=False):
        # Compute embeddings from all beats
        embeds = super(SSMnet, self).apply_cnn(x.transpose(0, 1)).unsqueeze(0)
        # Apply relative global attention
        # x.shape == (batch_size, seq_len, d_model)
        residual = self.identity(embeds)
        batch_size, seq_len, _ = embeds.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )
        q = self.selu(self.query(embeds))
        k = self.selu(self.key(embeds))
        v = self.selu(self.value(embeds))
        print(q.shape, k.shape, v.shape)
        # shape = (batch_size, seq_len, d_model)
        k_pos = self.pos(k).transpose(-1,-2)
        print(k_pos.shape)
        QK_t = torch.matmul(q , k_pos)
        # QK_t.shape = (batch_size, seq_len, seq_len)
        attn = QK_t / math.sqrt(q.size(-1))
        # attn.shape = (batch_size, seq_len, seq_len)
        #attn = torch.sigmoid(attn)
        attn = F.softmax(attn, dim=-1)
        out = F.normalize(torch.matmul(attn, v), p=2)
        # out.shape == (batch_size, seq_len, d_model)
        out += residual
        #out, _ = self.mh_att(q.transpose(0,1),k_pos.transpose(0,1),v.transpose(0,1))
        
        embeds_att = F.normalize(out, p=2)
        smm_hat = torch.cdist(embeds_att.transpose(0,1), embeds_att.transpose(0,1), 2)
        smm_hat = 1 - smm_hat/torch.max(smm_hat)
        
        return smm_hat
        


