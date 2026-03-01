import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataloaders(
    batch_size=64,
    data_dir="./data",
    num_workers=2
):
    """
    Returns train and test DataLoaders for MNIST dataset.
    """

    # Transform: convert image to tensor
    transform = transforms.ToTensor()

    # Training dataset
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Test dataset
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


# Optional test block (runs only when dataset.py is executed directly)
if __name__ == "__main__":
    train_loader, test_loader = get_mnist_dataloaders()

    images, labels = next(iter(train_loader))
    print("Train batch images shape:", images.shape)  # (B, 1, 28, 28)
    print("Train batch labels shape:", labels.shape)  # (B,)

