from datasets import load_dataset
from utils import show_images
from torchvision import transforms


def mnist_to_tensor(samples):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
    ])
    tensor_images = [transform(img) for img in samples]
    print(tensor_images)
    return tensor_images

def prepare_training_data():
    mnist = load_dataset("mnist")
    mnist = mnist.with_transform(mnist_to_tensor)
    mnist["train"] = mnist["train"].shuffle(seed=42)
    x = mnist["train"]["image"][0]
    print(x.min(), x.max(), x.shape)

    

def main():
    prepare_training_data()

if __name__ == "__main__":
    main()