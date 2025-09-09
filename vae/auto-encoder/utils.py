import matplotlib.pyplot as plt
import numpy as np


def show_images(images, titles=None, cols=4, figsize=(12, 8)):
    """
    Display a grid of images.
    
    Args:
        images: List or array of images to display
        titles: Optional list of titles for each image
        cols: Number of columns in the grid
        figsize: Figure size (width, height)
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols  # Calculate rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle case where we only have one row
    if rows == 1:
        axes = axes.reshape(1, -1) if n_images > 1 else [axes]
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    for i in range(n_images):
        # Convert PIL Image to numpy array if needed
        if hasattr(images[i], 'convert'):
            img = np.array(images[i])
        else:
            img = images[i]
        
        # Display the image
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Color image
            axes[i].imshow(img)
        else:
            # Grayscale image
            axes[i].imshow(img, cmap='gray')
        
        axes[i].axis('off')
        
        # Add title if provided
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
