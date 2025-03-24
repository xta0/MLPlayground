import torch
import numpy as np
from helper import convertToMPSImage

def saveToMPSImage(x):
    padding = torch.zeros(1,4,4)
    x       = torch.cat((x,padding),0) 
    # x       = torch.cat((x,padding),0) 
    # x       = torch.cat((x,padding),0) 
    x       = x.permute(1,2,0) #[RGBA,RGBA,RGBA,...]
    print("mps.x: ", x.shape)
    print(x)
    y       = torch.nn.functional.max_pool2d(x,(2,2))
    print("mps.y: ",y)
    x       = x.contiguous().view(-1).numpy()
    np.savetxt('./mp_x.txt',[x], delimiter=',')    
    return x

t = torch.randn([3,4,4])
saveToMPSImage(t)
print(t)
y = torch.nn.functional.max_pool2d(t,(2,2)) #kernel = [2,2], stride = [2,2]
padding = torch.zeros(1,2,2)
y = torch.cat((y,padding),0)
y = y.permute(1,2,0).contiguous().view(-1).detach().numpy()
print(y)