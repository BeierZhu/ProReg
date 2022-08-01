from .base_dataset import BaseDataset
import numpy as np

class NicoDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["nico_train"]
        elif split == "val":
            names = ["nico_val"]
        elif split == "test":
            names = ["nico_test"] 

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