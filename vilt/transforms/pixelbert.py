from .utils import (
    inception_normalize,
    MinMaxResize,
)
from torchvision import transforms
from .randaug import RandAugment

import torch
import clip
_, preprocess = clip.load('ViT-B/32', 'cpu')

def pixelbert_transform(size=800):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            preprocess,
        ]
    )


def pixelbert_transform_randaug(size=800):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            preprocess,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs
