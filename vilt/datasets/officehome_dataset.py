from .base_dataset import BaseDataset
import numpy as np

class OfficeHomeDataset(BaseDataset):
    def __init__(self, *args, split="", train_domain="", test_domain="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split


        if split == "train":
            names = [f'{train_domain}']
        elif split == "val":
            names = [f'{test_domain}']
        elif split == "test":
            names = [f'{test_domain}'] 
            
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