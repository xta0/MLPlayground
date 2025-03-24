## Implement a one hidden layer nerual network using Pytorch

import torch
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = torch.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = torch.softmax(self.output(x), dim=1)
        
        return x

# Create the network and look at it's text representation
model = Network()
output = model.forward(torch.ones(64,784))
print(output)

# save model to torchscript
# model.eval()
# example = torch.rand(64,784)
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("model.pt")


