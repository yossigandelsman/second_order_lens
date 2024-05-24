from typing import Any, Tuple
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder


class SubsampledImageNet(ImageNet):
    def __len__(self) -> int:
        return super().__len__() // 250
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index * 250)
    
class SubsampledValImageNet(ImageNet):
    def __len__(self) -> int:
        return super().__len__() // 10
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index * 10)