import os
from torchvision import datasets
import pickle as pkl
from PIL import Image

from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader

import os

import torchvision.datasets as datasets
import pytorch_lightning as pl
import torchvision.transforms as transforms


class DigitDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.dataset_name = config.root.split("/")[-1].split("_")[0]

        self.init_dataset()
        self.init_params()

    def train_dataloader(self):
        if self.config.model_name == "timm":
            return DataLoader(self.train_set_src, batch_size=self.config.batch_size,
                          shuffle=True, num_workers=self.config.num_workers)
        else:
            # if "EMNIST" in self.config.root_tgt_train:
            #     combined_option = "min_size"
            # else:
            #     combined_option = "max_size_cycle"
            combined_option = "max_size_cycle"
            return CombinedLoader({
                "src": DataLoader(self.train_set_src, batch_size=self.config.batch_size,
                                shuffle=True, pin_memory=True, num_workers=self.config.num_workers, drop_last=True),
                "tgt": DataLoader(self.train_set_tgt, batch_size=self.config.batch_size,
                                shuffle=True, pin_memory=True, num_workers=self.config.num_workers, drop_last=True),
            }, combined_option)

    def val_dataloader(self):
        if self.config.model_name == "timm":
            return DataLoader(self.val_set_src, batch_size=self.config.batch_size,
                              num_workers=self.config.num_workers)
        else:
            return DataLoader(self.val_set_tgt, batch_size=self.config.batch_size,
                              pin_memory=True, num_workers=self.config.num_workers, drop_last=True)

    def test_dataloader(self):
        dataloaders = [
            DataLoader(dataset, batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers) for dataset in self.test_sets]
        return dataloaders

    def init_dataset(self):
        transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.CenterCrop((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            ChanDup(),
        ])

        self.train_set_src = self.get_dataset(root=self.config.root, train=True, transform=transform)
        self.val_set_src = self.get_dataset(root=self.config.root, train=False, transform=transform)

        if self.config.model_name != "timm":
            assert self.config.root_tgt is not None

            # target dataset
            if self.config.root_tgt_train is None:
                self.train_set_tgt = self.get_dataset(root=self.config.root_tgt, train=True, transform=transform)
                self.val_set_tgt = self.train_set_tgt
            else:
                # Zero-shot evaluation setting
                self.train_set_tgt = self.get_dataset(root=self.config.root_tgt_train, train=True, transform=transform)
                self.val_set_tgt = self.get_dataset(root=self.config.root_tgt, train=True, transform=transform)

        self.test_sets = self.get_all_domain_datasets(self.config.root, transform)

    def init_params(self):
        if self.dataset_name == "MNIST":
            self.num_classes = 10
        elif self.dataset_name == "EMNIST":
            self.num_classes = 26
        elif self.dataset_name == "FashionMNIST":
            self.num_classes = 10

    def get_dataset(self, root, train, transform):
        dataset_domain_name = root.split("/")[-1]

        if dataset_domain_name == "MNIST_G":
            dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        elif dataset_domain_name == "FashionMNIST_G":
            dataset = datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
        elif dataset_domain_name == "EMNIST_G":
            dataset = datasets.EMNIST(root=root, train=train, split="letters", download=True, transform=transform)
        else:
            dataset = CENDataset(root=root, train=train, transform=transform)

        return dataset

    def get_all_domain_datasets(self, root, transform):
        dataset_name = self.dataset_name

        if dataset_name == "MNIST":
            return [
                datasets.MNIST(root=root.split("_")[0]+"_G", train=False, download=True, transform=transform),
                CENDataset(root=root.split("_")[0]+"_C", train=False, transform=transform),
                CENDataset(root=root.split("_")[0]+"_E", train=False, transform=transform),
                CENDataset(root=root.split("_")[0]+"_N", train=False, transform=transform),
            ]
        elif dataset_name == "EMNIST":
            return [
                datasets.EMNIST(root=root.split("_")[0]+"_G", train=False, split="letters", download=True, transform=transform),
                CENDataset(root=root.split("_")[0]+"_C", train=False, transform=transform),
                CENDataset(root=root.split("_")[0]+"_E", train=False, transform=transform),
                CENDataset(root=root.split("_")[0]+"_N", train=False, transform=transform),
            ]
        elif dataset_name == "FashionMNIST":
            return [
                datasets.FashionMNIST(root=root.split("_")[0]+"_G", train=False, download=True, transform=transform),
                CENDataset(root=root.split("_")[0]+"_C", train=False, transform=transform),
                CENDataset(root=root.split("_")[0]+"_E", train=False, transform=transform),
                CENDataset(root=root.split("_")[0]+"_N", train=False, transform=transform),
            ]


class CENDataset(datasets.VisionDataset):
    """
    Three transformed domain (Color, Edge, Negative) datasets
    from MNIST, FashionMNIST, NIST SD 19, EMNIST
    """

    def __init__(self, root, train=True, transform=None, target_transform=None):
        """Init GCENDataset dataset."""
        super(CENDataset, self).__init__(root=root)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError(
                f"{self.root.split('/')[-1]} Dataset not found. Use data gen script.")

        try:
            data = pkl.load(open(os.path.join(self.root, "data.pkl"), 'rb'))
        except ValueError:
            import pickle5
            data = pickle5.load(open(os.path.join(self.root, "data.pkl"), 'rb'))
        if train:
            self.data = data["train"]["data"]
            self.targets = data["train"]["targets"]
        else:
            self.data = data["test"]["data"]
            self.targets = data["test"]["targets"]

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
        index (int): Index
        Returns:
        tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, "data.pkl"))


class ChanDup:
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        if img.shape[0] == 1:
            return img.repeat(3, 1, 1)
        else:
            return img
