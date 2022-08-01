from .base_dataset import BaseDataset
import numpy as np

class PacsDataset(BaseDataset):
    def __init__(self, *args, split="", test_domain="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        domain_list = ['photo', 'art_painting', 'cartoon', 'sketch']
        domain_list.remove(test_domain)
        names = [] 
        if split == "train":
            names = [f'{domain}_train' for domain in domain_list]
        elif split == "val":
            names = [f'{domain}_crossval' for domain in domain_list]
        elif split == "test":
            names = [f'{test_domain}_test'] 
            
        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]

        index, question_index = self.index_mapper[index]

        label = self.table["label"][index].as_py()

        return {
            "image": image_tensor,
            "label": label,
        }