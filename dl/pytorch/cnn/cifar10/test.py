import torch
import torch.nn as nn
import torch.nn.functional as F

class MM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 128)
    def forward(self, x1, x2):
        for i in range(1, 10):
            print(i)
        if x2[0] > 0:
            x = self.fc(x1)
        else:
            x = F.relu(x1)
        return x

m = MM()


x1 = torch.rand(1, 1024)
x2 = -torch.rand(1)

m(x1, x2)

m = torch.jit.trace(m, (x1, x2))
# m = torch.jit.script(m, (x1, x2))
print(m.graph)