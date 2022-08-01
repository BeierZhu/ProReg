from .base_dataset import BaseDataset
import numpy as np

class DomainNetDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        train_domain = 'sketch'
        test_domain_list = ['clipart', 'painting', 'real']

        names = [] 
        if split == "train":
            names = [f'{train_domain}_train']
        elif split == "val":
            names = [f'{train_domain}_test']
        elif split == "test":
            names = [f'{test_domain}_test' for test_domain in test_domain_list] 
            
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