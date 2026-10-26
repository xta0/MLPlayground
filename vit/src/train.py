import torch
from vit.src.data import prepare_data

config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}

def main():
    batch_size = 32
    num_workers = 2
    trainloader, testloader, classes = prepare_data(batch_size, num_workers)

    # Example: Iterate through the training data
    for images, labels in trainloader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        break  # Just to demonstrate, we break after the first batch

if __name__ == "__main__":
    main()
    
    
    