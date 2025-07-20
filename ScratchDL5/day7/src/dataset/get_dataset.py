import torchvision
from torchvision import transforms

def get_train_dataset():
    preprocess = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root='/home/urp_jwl/.vscode-server/data/Diffusion/data', train=True, download=True, transform=preprocess)
    return train_dataset

if __name__ == "__main__":
    train_dataset = get_train_dataset()