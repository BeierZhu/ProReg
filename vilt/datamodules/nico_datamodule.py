from vilt.datasets import NicoDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class NicoDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return NicoDataset

    @property
    def dataset_name(self):
        return "nico"

    def setup(self, stage):
        super().setup(stage)
