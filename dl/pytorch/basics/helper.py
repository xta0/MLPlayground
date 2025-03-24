import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable


def test_network(net, trainloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels([
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
            'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
        ],
                            size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


# def verify(p1, p2, atol=0.0045):
def verify(p1, p2, atol=0.01):
    y1 = np.loadtxt(p1, delimiter=',')
    y2 = np.loadtxt(p2, delimiter=',')
    print(y1.shape)
    print(y2.shape)
    return np.allclose(y1, y2, atol=atol)


# input [C,H,W]
def convertToMPSImage(x):
    shape = x.shape
    # print("input shape: ", shape)
    slices = []
    n = int((shape[0] + 3) / 4)  # slices
    l = int(shape[0])
    d = l
    if l%4 != 0:
        d = (int(l / 4) + 1) * 4
    # print("(slice, c, dst_c)", n, l, d)
    for i in range(n):
        if i * 4 + 4 < l:
            s = x[i * 4:i * 4 + 4]
            slices.append(s)
        else:
            # add padding
            padding = torch.zeros(1, shape[1], shape[2])
            s = x[i * 4:shape[0]]
            while l < d:
                print("concating")
                s = torch.cat((s, padding), 0)
                l = l + 1
            slices.append(s)
    # print(slices)
    # flatten the numpy array
    slices = [x.permute(1, 2, 0).contiguous().view(-1).detach().numpy() for x in slices]
    slices = np.concatenate(slices)
    return slices
