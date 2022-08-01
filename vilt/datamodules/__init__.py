from .bar_datamodule import BarDataModule
from .nico_datamodule import NicoDataModule
from .imagenetLT_datamodule import ImageNetLTDataModule
from .pacs_datamodule import PacsDataModule
from .officehome_datamodule import OfficeHomeDataModule
from .domain_net_datamodule import DomainNetDataModule

_datamodules = {
    "bar": BarDataModule,
    "nico": NicoDataModule,
    "imagenetLT": ImageNetLTDataModule,
    'pacs': PacsDataModule,
    'officehome': OfficeHomeDataModule,
    'domain_net': DomainNetDataModule
}
