import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


class CifarDataset(CIFAR10):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        image = transforms.ToTensor()(image)
        batch = dict(image=image, label=label)
        return batch
