import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipyplot
from PIL import Image

img_path = './mona.jpg'
img = plt.imread(img_path)
print(img.shape) # ndarray
print(img.dtype) # uint8 [0-255]
print(img[24, 24]) # RGB [186 180 128]
# plt.imshow(img)
# plt.show()

# normalze the image
img = img.astype(np.float32) / 255.0
print(img[24, 24]) # RGB [186 180 128]
# plt.show()
# parameters
num_iteration = 16
beta = 0.1

images = []
steps = ["Step:" + str(i) for i in range(num_iteration)]

# forward diffusion
for i in range(num_iteration):
    mean = np.sqrt(1-beta) * img
    img = np.random.normal(mean, beta, img.shape)
    pil_img = (img*255).astype('uint8')
    images.append(pil_img)

# Select the first image, every second image, and then ensure the last is included.
# For 16 images, this produces indices: 0, 2, 4, 6, 8, 10, 12, 15.
selected_indices = [0, 2, 4, 6, 8, 10, 12, len(images) - 1]

# Grab the selected images and labels
selected_images = [images[i] for i in selected_indices]
selected_labels = [steps[i] for i in selected_indices]

# Create a figure with 1 row and 8 columns
fig, axs = plt.subplots(1, 8, figsize=(16, 2))  # Adjust figsize as needed

# Loop through each selected image and display it with its label
for ax, img_array, label in zip(axs, selected_images, selected_labels):
    ax.imshow(img_array)
    ax.set_title(label, fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()