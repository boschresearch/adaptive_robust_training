# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from typing import Any, Dict, Optional, Tuple

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class CIFAR10DataModule(LightningDataModule):
    """Example of LightningDataModule for CIFAR10 dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_split: Tuple[int, int] = (45_000, 5_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_mean: Tuple[float] = (0.4914, 0.4822, 0.4465),
        data_std: Tuple[float] = (0.2023, 0.1994, 0.2010),
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        # data transformations
        normalize = transforms.Normalize(mean=data_mean, std=data_std)
        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, 4),
                transforms.RandomCrop(32, 2, padding_mode="edge"),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.test_transforms = transforms.Compose([transforms.ToTensor(), normalize])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = datasets.CIFAR10(
                self.hparams.data_dir, train=True, transform=self.transforms
            )
            self.data_val = datasets.CIFAR10(
                self.hparams.data_dir, train=False, transform=self.test_transforms
            )
            self.data_test = datasets.CIFAR10(
                self.hparams.data_dir, train=False, transform=self.test_transforms
            )
            # train_dataset_size = len(trainset)
            # indices = list(range(train_dataset_size))
            # train_split = self.hparams.train_val_split[0]
            # np.random.seed(42)
            # np.random.shuffle(indices)
            # train_indices, val_indices = indices[:train_split], indices[train_split:]
            # assert len(val_indices) == self.hparams.train_val_split[1]

            # self.data_train = Subset(trainset, train_indices)
            # self.data_val = Subset(valset, val_indices)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "cifar10.yaml")
    # cfg.data_dir = str(root / "data")
    dataset = hydra.utils.instantiate(cfg)
    dataset.prepare_data()
    dataset.setup()
    print(len(dataset.data_train))
