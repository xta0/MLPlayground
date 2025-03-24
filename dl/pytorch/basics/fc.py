import torch 
import torch.nn as nn
import numpy as np
from helper import convertToMPSImage

def saveToMPSImage(x):
    # x = x.unsqueeze(0)
    # x = x.unsqueeze(0)
    # print("saveToMPS:",x.shape)
    # padding = torch.zeros(1,1,3)
    # x       = torch.cat((x,padding),0) 
    # x       = torch.cat((x,padding),0) 
    # x       = torch.cat((x,padding),0) 
    # x       = x.permute(1,2,0) #[32,32,4] [RGBA,RGBA,RGBA,...]
    # print("mps-x.shape: ", x.shape)
    x       = x.contiguous().view(-1).numpy()
    np.savetxt('./fc_x.txt',[x], delimiter=',')    
    return x

model = torch.nn.Sequential(
    torch.nn.Linear(16, 8)
)
for m in model.modules():
    if isinstance(m, nn.Linear):
        # m.weight = nn.Parameter(torch.ones([8,18]))
        w = m.weight.view(-1).detach().numpy()
        np.savetxt('./fc_W.txt',[w], delimiter=',')
        b = m.bias.view(-1).detach().numpy()
        print(b.shape)
        np.savetxt('./fc_b.txt',[b], delimiter=',')
        

# x = torch.rand([3,3,3])
#saveToMPSImage(x)
# x = torch.tensor([1.0,2.0,3.0])
x = torch.rand([1,18])
# x = x.view(-1)
print(x.shape)
saveToMPSImage(x)
y = model(x)
print(y.shape)
y = y.view(-1).detach().numpy()
print(y)
