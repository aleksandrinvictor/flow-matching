from typing import Callable, Optional, Dict

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST


class FlowMatchingMNIST(Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None):
        self.dataset = MNIST(root=root, train=train, transform=transform, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image, label = self.dataset[index]

        return {"image": image}