import os
import numpy as np
import torch
from torchvision import datasets
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

# check if CUDA is available
use_cuda = torch.cuda.is_available()

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 20
num_workers = 0
transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transformer2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('./dogImages/train', transform=transformer)
valid_data = datasets.ImageFolder('./dogImages/test', transform=transformer2)
test_data = datasets.ImageFolder('./dogImages/valid', transform=transformer2)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=16,
                                          num_workers=num_workers,
                                          shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=16,
                                           num_workers=num_workers,
                                           shuffle=True)
loaders_transfer = {
    "train": train_loader,
    "valid": valid_loader,
    "test": test_loader
}

model_transfer = models.vgg16(pretrained=True)
print(model_transfer)

# freeze the parameters
for param in model_transfer.features.parameters():
    param.requires_grad = False

# replace the last layer
model_transfer.classifier[6] = nn.Linear(
    model_transfer.classifier[6].in_features, 133, bias=True)

print(model_transfer)

if use_cuda:
    model_transfer = model_transfer.cuda()

criterion_transfer = torch.nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.parameters(), lr=0.01)


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            Y = model(data)
            loss = criterion(Y, target)
            loss.backward()
            optimizer.step()
            #             train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            Y = model(data)
            loss = criterion(Y, target)
            #             valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            valid_loss += loss.item() * data.size(0)

        # calculate the average loss
        train_loss = train_loss / len(loaders['train'].dataset)
        valid_loss = valid_loss / len(loaders['valid'].dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.
              format(epoch, train_loss, valid_loss))

        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(
                'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                .format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

    # return trained model
    return model


# train the model
print("use_cuda: ", use_cuda)
model_transfer = train(10, loaders_transfer, model_transfer,
                       optimizer_transfer, criterion_transfer, use_cuda,
                       'model_transfer.pt')

# load the model that got the best validation accuracy
model_transfer.load_state_dict(torch.load('model_transfer.pt'))

# test
def test(loaders, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):

        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) *
                                 (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(
            np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' %
          (100. * correct / total, correct, total))


# call test function
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)