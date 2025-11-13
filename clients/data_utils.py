import torch
from torchvision import datasets, transforms

def get_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

    return train_loader, test_loader
