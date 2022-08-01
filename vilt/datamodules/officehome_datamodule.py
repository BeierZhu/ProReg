from vilt.datasets import OfficeHomeDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class OfficeHomeDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return OfficeHomeDataset

    @property
    def dataset_name(self):
        return "officehome"

    def setup(self, stage):
        super().setup(stage)
