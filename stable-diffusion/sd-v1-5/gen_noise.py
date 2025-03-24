import numpy as np
import matplotlib.pyplot as plt

# 1. Generate a (300, 300, 3) array of Gaussian noise
noise_rgb = np.random.normal(loc=0.0, scale=1.0, size=(300, 300, 3))

# 2. Rescale/normalize the noise to [0, 1] for display
#    (Min-max normalization: (x - min) / (max - min))
min_val = noise_rgb.min()
max_val = noise_rgb.max()
noise_rgb_norm = (noise_rgb - min_val) / (max_val - min_val)

# 3. Display the noise image
plt.imshow(noise_rgb_norm)
plt.title("")
plt.axis('off')
plt.show()