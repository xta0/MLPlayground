import numpy as np
import matplotlib.pyplot as plt

# Load the sprite data from the .npy file
sprites = np.load('./sprites_1788_16x16.npy')
print("Sprites shape:", sprites.shape)  # e.g., (89400, 16, 16, 3) or (89400, 16, 16)

# Randomly select 60 sprites
num_samples = 60
sample_indices = np.random.choice(sprites.shape[0], num_samples, replace=False)
sample_sprites = sprites[sample_indices]

# Alternatively, if you want the first 60 sprites, you can use:
# sample_sprites = sprites[:60]

# Determine grid dimensions; here we use 10 columns and 6 rows (since 10 * 6 = 60)
cols = 10
rows = 6

fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
axes = axes.flatten()

# Loop over the sampled sprites and display each one
for i, sprite in enumerate(sample_sprites):
    # If the sprite is grayscale (has no channel or one channel), use cmap='gray'
    if sprite.ndim == 2 or (sprite.ndim == 3 and sprite.shape[-1] == 1):
        axes[i].imshow(sprite.squeeze(), cmap='gray')
    else:
        axes[i].imshow(sprite)
    axes[i].axis('off')
    axes[i].set_title(f"{i}", fontsize=8)

plt.tight_layout()
plt.show()