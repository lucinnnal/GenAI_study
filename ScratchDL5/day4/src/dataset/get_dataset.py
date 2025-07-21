import torch
import torchvision
from torchvision import transforms

def get_train_dataset():
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(torch.flatten) # Flatten the image to 1D vector
        ])
    train_dataset = torchvision.datasets.MNIST(root='/home/urp_jwl/.vscode-server/data/VAE/data', train=True, download=True, transform=preprocess)
    return train_dataset

if __name__ == "__main__":
    train_dataset = get_train_dataset()