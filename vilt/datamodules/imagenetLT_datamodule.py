from vilt.datasets import ImageNetLTDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class ImageNetLTDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ImageNetLTDataset

    @property
    def dataset_name(self):
        return "imagenetLT"

    def setup(self, stage):
        super().setup(stage)
