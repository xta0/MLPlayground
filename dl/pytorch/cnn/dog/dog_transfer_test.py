import numpy as np
import torch
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# check if CUDA is available
use_cuda = torch.cuda.is_available()
print("use_cuda: ", use_cuda)

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

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=16,
                                          num_workers=0,
                                          shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=16,
                                           num_workers=0,
                                           shuffle=True)
loaders_transfer = {"valid": valid_loader, "test": test_loader}

model_transfer = models.vgg16(pretrained=True)
print(model_transfer)

# freeze the parameters
for param in model_transfer.features.parameters():
    param.requires_grad = False

# replace the last layer
model_transfer.classifier[6] = nn.Linear(
    model_transfer.classifier[6].in_features, 133, bias=True)

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
test(loaders_transfer, model_transfer, torch.nn.CrossEntropyLoss(), use_cuda)

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in train_data.classes]
print("class_names: ", class_names)


def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    img = Image.open(img_path).convert('RGB')
    # resize and normalze
    img = transformer2(img).unsqueeze(0)
    output = model_transfer(img).view(-1).contiguous()
    index = torch.argmax(output)
    return class_names[index]

