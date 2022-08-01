from vilt.datasets import PacsDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class PacsDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return PacsDataset

    @property
    def dataset_name(self):
        return "pacs"

    def setup(self, stage):
        super().setup(stage)
