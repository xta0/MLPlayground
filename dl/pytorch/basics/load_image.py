import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
import helper

"""
structures:

root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png

resources:
https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip
"""

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
dataset    = datasets.ImageFolder('./images', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True) #python generator
