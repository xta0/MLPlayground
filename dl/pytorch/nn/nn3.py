## Implement a one hidden layer nerual network using Pytorch

import torch
from torch import nn

import pandas as pd
df = pd.DataFrame([[10, 20, 30], [100, 200, 300]],
                  columns=['foo', 'bar', 'baz'])
def get_methods(object, spacing=20):
  methodList = []
  for method_name in dir(object):
    try:
        if callable(getattr(object, method_name)):
            methodList.append(str(method_name))
    except:
        methodList.append(str(method_name))
  processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
  for method in methodList:
    try:
        print(str(method.ljust(spacing)) + ' ' +
              processFunc(str(getattr(object, method).__doc__)[0:90]))
    except:
        print(method.ljust(spacing) + ' ' + ' getattr() failed')

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 4)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        return x

# Create the network and look at it's text representation
model = Network()
model.eval()
state = model.state_dict()
print("state: ", state)
x = torch.rand(1,4)
y1 = model(x)
ts = torch.jit.trace(model, x)
y2 = ts(x)
print(y1)
print(y2)

torch.jit.save(ts, './nn3.pt')
ts._save_for_lite_interpreter('./nn3.ptl')

# torch graph
print(ts.forward.graph)

# deserilize using pikle

import pickle

with open('data.bin', 'rb') as f:
    data = pickle.load(f)

    print(data)