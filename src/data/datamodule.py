import numpy as np
import pytorch_lightning as pl

from typing import Optional, Tuple
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

class AnimalDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        image_dir : str,
        train_val_test_split: Tuple[float, float, float] = (0.75, 0.15, 0.10),
        batch_size : int = 32,
        num_workers : int = 0,
        pin_memory : bool = False,
        **kwargs
    ):
        """__init__ method

        Load possible configurations for data loader with a selected image
        transforms methods

        Args:
        - image_dir: str
              Directory containing images in different class-folder.
        - train_val_test_split: Tuple[float,float,float] = (0.75, 0.15, 0.10)
              Tuple describing fractions for each training/validation/testing data
              (Default = (0.75, 0.15, 0.10) or 75% training data, 15% validation
              data, 10% testing data).
        - batch_size: int = 32
              Number of sample in each batch (Default = 32)
              Please reduce this parameter if 'Out Of Memory' is reached.
        - num workers: int = 0
              How many subprocesses to use for data loading. 
              0 means that the data will be loaded in the main process. 
              (Default = 0)
        - pin_memory: bool = False
              If True, the data loader will copy Tensors into CUDA pinned 
              memory before returning them.

        Reference:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

        """
        super().__init__()

        self.image_dir = image_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.TRAIN_TRANSFORM = transforms.Compose(
            [
                transforms.Resize((224, 224)),,
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation((0, 90))
                transforms.ToTensor(),
            ]
        )

        self.VAL_TRANSFORM = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val_test: Optional[Dataset] = None
    
    def setup(self, stage: Optional[str] = None):
        self.data_train = ImageFolder(
            root = self.image_dir, transform = self.TRAIN_TRANSFORM)
        self.data_val_test = ImageFolder(
            root = self.image_dir, transform = self.VAL_TRANSFORM
        )    

        train_val_test = tuple(
            [i * len(self.data_train) for i in self.train_val_test_split]
        )

        valid_size = train_val_test[1]
        test_size = train_val_test[2]
        valid_test_size = valid_size + test_size 
        num_train = len(self.data_train)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(valid_test_size * num_train))
        vt_split = int(np.floor(test_size * num_train))
        train_idx, vt_idx = indices[split:], indices[:split]
        val_idx, test_idx = vt_idx[vt_split:], vt_idx[:vt_split]

        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(val_idx)
        self.test_sampler = SubsetRandomSampler(test_idx)

    def train_dataloader(self):
        return DataLoader(
            dataset = self.data_train,
            sampler = self.train_sampler,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset = self.data_val_test,
            sampler = self.valid_sampler,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset = self.data_val_test,
            sampler = self.test_sampler
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
        )


