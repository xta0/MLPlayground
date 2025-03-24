import torch
import torch.nn as nn
import numpy as np
from helper import convertToMPSImage


def conv2d_s1p1():
    x = np.loadtxt('./tests/x.txt', delimiter=',')
    x = torch.from_numpy(x).view(3, 32, 32)
    # x = torch.rand([3, 32, 32])
    # mps_x = convertToMPSImage(x)
    # print(mps_x.shape)
    # np.savetxt('./tests/mps_x.txt', [mps_x], delimiter=',')
    x = x.unsqueeze(0)
    print(x.shape)
    conv = nn.Conv2d(3, 16, 3, padding=1)  # ic, oc, kernel size, padding
    with torch.no_grad():
        w = np.loadtxt('./tests/W.txt', delimiter=',')
        w = torch.from_numpy(w).view(16, 3, 3, 3)
        print(w.dtype)
        b = np.loadtxt('./tests/b.txt', delimiter=',')
        b = torch.from_numpy(b).view(16)
        conv.weight = nn.Parameter(w)
        conv.bias = nn.Parameter(b)

    w = conv.weight
    # padding = torch.zeros(1,3,3,3)
    # w       = torch.cat((w,padding),0)
    # w       = torch.cat((w,padding),0)
    print(w.shape)
    b = conv.bias
    print(b.shape)

    # w = w.permute(0, 2, 3, 1).contiguous().view(-1).detach().numpy()
    # np.savetxt('./tests/mps_W.txt', [w], delimiter=',')

    # b = b.view(-1).detach().numpy()
    # np.savetxt('./tests/mps_b.txt', [b], delimiter=',')

    # y       = torch.sigmoid(conv(x))
    y = conv(x)
    print(y.shape)
    yy = y.view(-1).detach().numpy()
    np.savetxt('./tests/nchw_y.txt', [yy], delimiter=',')
    # print(y)
    y = y.squeeze(0)
    mps_y = convertToMPSImage(y)
    print(mps_y.shape)
    np.savetxt('./tests/nhwc_y.txt', [mps_y], delimiter=',')


conv2d_s1p1()