
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torchvision
from torchvision import transforms

class ImageDataset(Dataset):
    """Image Dataset that works with images

    This class inherits from torch.utils.data.Dataset and will be used inside torch.utils.data.DataLoader
    Args:
        data (str): Dataframe with path and label of images.
        transform (torchvision.transforms.Compose, optional): Transform to be applied on a sample. Defaults to None.

    Examples:
        >>> df, train_df, test_df = create_and_load_meta_csv_df(dataset_path, destination_path, randomize=randomize, split=0.99)
        >>> train_dataset = dataset.ImageDataset(train_df)
        >>> test_dataset = dataset.ImageDataset(test_df, transform=...)
    """

    def __init__(self, data, transform=None):
        self.data = data
        self.transform =torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data
        image = Image.open(img_path)# load PIL image



        if self.transform:
            image = self.transform(image)

        return image
