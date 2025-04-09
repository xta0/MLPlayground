import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import accumulate
import os

# --- Step 1: Read and Normalize the Image ---
img_path = './mona.jpg'
img = plt.imread(img_path)

# Normalize: convert from [0, 255] to [0, 1] and then to [-1, 1]
img = img.astype(np.float32) / 255.0
img = img * 2 - 1  # [0, 1] -> [-1, 1]

# --- Step 2: Set Up the Diffusion Parameters ---
num_iteration = 16
betas = np.linspace(0.0001, 0.02, num_iteration)
alpha_list = [1 - beta for beta in betas]
alpha_bar_list = list(accumulate(alpha_list, lambda x, y: x * y))

# --- Step 3: Compute x_t ---
selected_indices = list(range(0, num_iteration, 2))
images = []

# Create output folder if it doesn't exist
os.makedirs("output_images", exist_ok=True)

for t in selected_indices:
    x_t = (np.sqrt(1 - alpha_bar_list[t]) * np.random.normal(0, 1, img.shape) +
           np.sqrt(alpha_bar_list[t]) * img)

    # Restore x_target from [-1,1] back to [0,1]
    x_t = (x_t + 1) / 2
    x_t = np.clip(x_t, 0, 1)  # Ensure valid range
    x_t_uint8 = (x_t * 255).astype('uint8')
    images.append(x_t_uint8)

    # --- Save individual image without border ---
    im = Image.fromarray(x_t_uint8)
    im.save(f"output_images/noisy_t={t}.png")

# --- Step 4: Display and Save the Combined Figure ---
fig, axs = plt.subplots(1, len(images), figsize=(20, 3))
for ax, x_img, t in zip(axs, images, selected_indices):
    ax.imshow(x_img)
    ax.set_title(f"t={t}")
    ax.axis('off')

plt.tight_layout()
fig.savefig("diffusion_grid.png", bbox_inches='tight', pad_inches=0)
# plt.show()
