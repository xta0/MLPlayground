import numpy as np
import matplotlib.pyplot as plt

# 1. Generate a (300, 300, 3) array of Gaussian noise
noise_rgb = np.random.normal(loc=0.0, scale=1.0, size=(300, 300, 3))

# 2. Normalize the noise to [0, 1] for display
min_val = noise_rgb.min()
max_val = noise_rgb.max()
noise_rgb_norm = (noise_rgb - min_val) / (max_val - min_val)

# 3. Display the noise image without white borders
plt.imshow(noise_rgb_norm)
plt.axis('off')
plt.gca().set_position([0, 0, 1, 1])           # Remove axis padding
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove figure padding
# plt.show()

# 4. Save the image to disk without borders
plt.imsave('noise.png', noise_rgb_norm)  # Saves just the image data, no padding
