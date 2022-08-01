from .base_dataset import BaseDataset
import numpy as np

class BarDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["bar_train"]
        elif split == "val":
            names = ["bar_val"]
        elif split == "test":
            names = ["bar_test"] 

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

        if self.split != "test":
            label = self.table["label"][index].as_py()
        else:
            label = list()

        return {
            "image": image_tensor,
            "label": label,
        }