import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import sys
sys.path.append('../')
import helper

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Construct NN
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        # self.fc2 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(256, 10)
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        # output so no dropout here
        x = F.log_softmax(self.fc2(x),dim=1)

        return x

model = Classifier()
model.load_state_dict(torch.load('model_mnist.pt'))
model.eval()
print(model)

# Raw dataset
rawset = datasets.MNIST('./MNIST_data/')
cnt = 64
cn  = 0
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
with torch.no_grad():
    for idx, (img, label) in enumerate(rawset):
        if idx < cnt:
            t = transform(img) #[1,28,28]
            t = t.view(t.shape[0],-1) # [1, 784]
            output = model(t)
            ps = torch.exp(output)
            p, clz = ps.topk(1, dim=1)
            clz = clz.view(-1).item()
            if clz == label:
                print(f"True: ( output:{clz}, label:{label} )")
                cn += 1
            else:
                print(f"False: ( output:{clz}, label:{label} )")
    print(f"accuracy: {float(cn)/float(cnt)}")

