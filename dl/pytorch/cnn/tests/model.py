import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x # skip connection


class Net(nn.Module):
    def __init__(self, n_chans=32, n_blocks=10):
        super(Net, self).__init__()
        self.n_chans = n_chans
        self.conv1 = nn.conv2d(3, n_chans, kernel_size=3, padding=1)
        # nn.Sequential(*args) takes an array
        self.resblocks = nn.Sequential(*[ResBlock(n_chans)] * n_blocks)
        self.fc1 = nn.Linear(8 * 8 * n_chans, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans1)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net(n_chans1=32, n_blocks=100)