import time
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

# get the "features" portion of VGG19 (we will not need the "classifier" portion)
vgg = models.vgg19(pretrained=True).features

# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)


def load_image(img_path, max_size=612, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    image = Image.open(img_path).convert('RGB')
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image


def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


content = load_image("./ian-liu.jpeg").to(device)
style = load_image("./starry_night_van_gogh.jpg",
                   shape=content.shape[-2:]).to(device)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# content and style ims side-by-side
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))
fig.savefig('before_train.png')


def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  #content representation
            '28': 'conv5_1',
        }
    features = {}
    # model._modules is a dictionary holding each module in the model
    # feed forward image to the model, collect the result features for the above layers
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    _, c, h, w = tensor.size()  #nchw
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

style_grams = {}
for layer in style_features:
    gram = gram_matrix(style_features[layer])
    style_grams[layer] = gram

# create a third "target" image and prep it for change
# it is a good idea to start off with the target as a copy of our *content* image
# then iteratively change its style
target = content.clone().requires_grad_(True).to(device)

# Loss and Weights
alpha = 1
beta = 1e6
style_weights = {
    'conv1_1': 0.2,
    'conv2_1': 0.2,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2
}

# Perform Gradient Descent
show_every = 400
optimizer = optim.Adam([target], lr=0.003)
steps = 2000
total_losses = []
t = time.time()
for ii in range(1, steps + 1):
    t1 = time.time()
    target_features = get_features(target, vgg)
    #content loss
    content_loss = torch.mean(
        (target_features['conv4_2'] - content_features['conv4_2'])**2)
    #style loss
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, c, h, w = target_feature.shape
        style_gram = style_grams[layer]
        layer_style_loss = torch.mean(
            (target_gram - style_gram)**2) / ((2 * c * h * w)**2)
        style_loss += style_weights[layer] * layer_style_loss
    #total loss
    total_loss = alpha * content_loss + beta * style_loss
    total_losses.append(total_loss)

    #update target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(im_convert(content))
        ax2.imshow(im_convert(target))
        fig.savefig(f"{ii}.png")

    t2 = time.time() - t1
    print(f"epoch: {ii}, time:{t2}, total_loss:{total_loss}")

training_time = time.time() - t
print(f"Done. Training time: {training_time}")

# draw total_loss
plt.plot(total_losses, label='Training loss')
plt.legend(frameon=False)
fig = plt.figure()
fig.savefig("train_loss.png")
