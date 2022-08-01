from vilt.datasets import DomainNetDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class DomainNetDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return DomainNetDataset

    @property
    def dataset_name(self):
        return "domain_net"

    def setup(self, stage):
        super().setup(stage)
