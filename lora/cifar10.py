# https://rocm.blogs.amd.com/artificial-intelligence/lora-fundamentals/README.html
# https://www.youtube.com/watch?v=fhIGt7QGg4w&ab_channel=koiboi

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.parametrize as parametrize

batch_size = 8

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
])
data_path = './dataset/'
train_data = datasets.CIFAR10(data_path,
                              train=True,
                              download=True,
                              transform=transform)
test_data = datasets.CIFAR10(data_path,
                             train=False,
                             download=True,
                             transform=transform)

num_train = len(train_data)  #50000
indices = list(range(num_train))
np.random.shuffle(indices)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# specify the image classes
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]

# Define the device
device = torch.device("mps" )

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3*32*32, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x



# helper function to display image
def image_display(images):
    # get the original image
    images = images * 0.5 + 0.5
    plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.show()


def train(model, num_epochs=1):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # loss_fn = nn.NLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.003)

    for epoch in range(num_epochs):  # training loop
        loss_log = 0.0
        i = 0
        for images, labels in train_loader:
            # inputs, labels = data[0].to(device), data[1].to(device)
            images, labels = images.to(device), labels.to(device)
            # Resets the parameter gradients
            optimizer.zero_grad()
    
            outputs = model(images)
            # print("output: ", outputs)
            # print("labels: ", labels)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print loss after every 1000 mini-batches
            loss_log += loss.item()
            if i % 2000 == 1999:    
                print(f'[{epoch}, {i+1:5d}] loss: {loss_log / 2000:.3f}')
                loss_log = 0.0
            i+=1

def test(model_path):
    # Prepare the test data.
    images, labels = next(iter(test_loader))
    # display the test images
    image_display(torchvision.utils.make_grid(images))
    # show ground truth labels
    print('Ground truth labels: ', ' '.join(f'{classes[labels[j]]}' for j in range(images.shape[0])))

    # Load the saved model and have a test
    model = Classifier()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]}'
                                for j in range(images.shape[0])))
    
def test2(model_path):
    model = Classifier()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # images = images.to(device)
            # labels = labels.to(device)
            # inference
            outputs = model(images)
            # get the best prediction
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the given model on the {total} test images is {100 * correct // total} %')

def test3(model_path):
    device = torch.device("mps")
    model = Classifier()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    apply_lora(model, device)
    enable_lora(model, enabled=True)
    # model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # images = images.to(device)
            # labels = labels.to(device)
            # inference
            outputs = model(images)
            # get the best prediction
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the given model on the {total} test images is {100 * correct // total} %')


def test4(model):
    enable_lora(model, enabled=True)
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # images = images.to(device)
            # labels = labels.to(device)
            # inference
            outputs = model(images)
            # get the best prediction
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    print(f'Accuracy of the given model on the {total} test images is {100 * correct // total} %')

def test5(model):
    enable_lora(model, enabled=True)
    model = model.to(device)
    # Prepare the test data.
    images, labels = next(iter(test_loader))
    # display the test images
    image_display(torchvision.utils.make_grid(images))
    # show ground truth labels
    print('Ground truth labels: ', ' '.join(f'{classes[labels[j]]}' for j in range(images.shape[0])))

    # Load the saved model and have a test
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]}'
                                for j in range(images.shape[0])))

class ParametrizationWithLoRA(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device='cpu'):
        super().__init__()

        # Create A B and scale used in ∆W = BA x α/r
        self.lora_weights_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
        nn.init.normal_(self.lora_weights_A, mean=0, std=1)
        self.lora_weights_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))

        # convert scale to device type
        # self.scale = torch.tensor(alpha / rank, dtype=torch.float32, device=device)
        self.scale = 1.0
        # self.scale = 1
        
        self.enabled = True

    def forward(self, original_weights):
        # if self.enabled:
        return original_weights + torch.matmul(self.lora_weights_B, self.lora_weights_A).view(original_weights.shape) * self.scale
        # else:
            # return original_weights

def apply_parameterization_lora(layer, device, rank=4, alpha=1):
    """
    Apply loRA to a given layer
    """
    features_in, features_out = layer.weight.shape
    return ParametrizationWithLoRA(
        features_in, features_out, rank=rank, alpha=alpha, device=device
    )
    
def enable_lora(model, enabled=True):
    """
    enabled = True: incorporate the the lora parameters to the model
    enabled = False: the lora parameters have no impact on the model
    """
    for layer in [model.fc1, model.fc2, model.fc3]:
        print(layer)
        layer.parametrizations["weight"][0].enabled = enabled

def apply_lora(model, device):
    parametrize.register_parametrization(model.fc1, "weight", apply_parameterization_lora(model.fc1, device))
    parametrize.register_parametrization(model.fc2, "weight", apply_parameterization_lora(model.fc2, device))
    parametrize.register_parametrization(model.fc3, "weight", apply_parameterization_lora(model.fc3, device))

def calculate_lora_params(model_path):
    device = torch.device("mps")
    model = Classifier()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    apply_lora(model, device)
    enable_lora(model, enabled=True)
    total_lora_params = 0
    total_original_params = 0
    for index, layer in enumerate([model.fc1, model.fc2, model.fc3]):
        total_lora_params += layer.parametrizations["weight"][0].lora_weights_A.nelement() + layer.parametrizations["weight"][0].lora_weights_B.nelement()
        total_original_params += layer.weight.nelement() + layer.bias.nelement()

    print(f'Number of parameters in the model with LoRA: {total_lora_params + total_original_params:,}')
    print(f'Parameters added by LoRA: {total_lora_params:,}')
    params_increment = (total_lora_params / total_original_params) * 100
    print(f'Parameters increment: {params_increment:.3f}%')

def train_lora(model_path, num_epochs):
    device = torch.device("mps")
    model = Classifier()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    apply_lora(model, device)
    enable_lora(model, enabled=True)
    # model.eval()
    for name, param in model.named_parameters():
        print("name: ", name)
        if 'lora' not in name:
            param.requires_grad = False
    train(model, num_epochs)
    model_path = './classifier_cifar10_lora.pth'
    torch.save(model.state_dict(), model_path)

    test4(model)
    test5(model)


def main():
   # get a batch of images
    # images, labels = next(iter(train_loader))
    # display images
    # image_display(torchvision.utils.make_grid(images))
    # show ground truth labels
    # print('Ground truth labels: ', ' '.join(f'{classes[labels[j]]}' for j in range(images.shape[0])))
    # ship ship airplane ship horse ship automobile automobile
    model = Classifier()
    model.to(device)
    print(model)
    # train the model for only 1 epoch
    # train(model, 2)
    # save the model
    model_path = './classifier_cifar10.pth'
    # torch.save(model.state_dict(), model_path)
    # run the model on test set
    # test(model_path)
    # test2(model_path)
    # Accuracy of the given model on the 10000 test images is 47 %
    # run lora
    # test3(model_path)
    # Accuracy of the given model on the 10000 test images is 47 %
    # calculate_lora_params(model_path)
    """
    Number of parameters in the model with LoRA: 21,059,634
    Parameters added by LoRA: 61,480
    Parameters increment: 0.293%
    """
    train_lora(model_path, 2)
    



# Ground truth labels:  cat ship ship airplane frog frog automobile frog
# Predicted:            cat automobile ship airplane deer frog cat frog
"""
[0,  2000] loss: 2.000
[0,  4000] loss: 1.725
[0,  6000] loss: 1.590
"""

if __name__ == "__main__":
    main()