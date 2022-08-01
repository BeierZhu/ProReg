from vilt.datasets import BarDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class BarDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return BarDataset

    @property
    def dataset_name(self):
        return "bar"

    def setup(self, stage):
        super().setup(stage)
