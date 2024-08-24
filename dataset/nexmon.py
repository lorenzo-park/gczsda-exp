from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from glob import glob

import cv2
import os

import albumentations as A
import torchvision.datasets as datasets
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms


def get_all_nexmon_datasets(root_src, fold_no, transform, classes=None, channels=3):
    roots = [
        "/shared/lorenzo/data-tubuki-cache/exp1-cwt",
        "/shared/lorenzo/data-tubuki-cache/exp2-cwt",
        "/shared/lorenzo/data-tubuki-cache/exp3-cwt",
        "/shared/lorenzo/data-tubuki-cache/exp4-cwt",
        "/shared/lorenzo/data-tubuki-cache/exp5-cwt",
        "/shared/lorenzo/data-tubuki-cache/exp6-cwt",
        "/shared/lorenzo/data-tubuki-cache/exp7-cwt",
        "/shared/lorenzo/data-tubuki-cache/exp8-cwt",
    ]
    return [
        NexmonDataset(
            root=roots[i],
            fold=fold_no if roots[i]==root_src else None,
            train=False,
            transform=transform,
            classes=classes,
            channels=channels,
        ) for i in range(len(roots))
    ]


class NexmonDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.init_dataset()
        self.init_params()

    def train_dataloader(self):
        if self.config.model_name == "timm":
            return DataLoader(self.train_set_src, batch_size=self.config.batch_size,
                          shuffle=True, num_workers=self.config.num_workers)
        else:
            return CombinedLoader({
                "src": DataLoader(self.train_set_src, batch_size=self.config.batch_size,
                                shuffle=True, pin_memory=True, num_workers=self.config.num_workers, drop_last=True),
                "tgt": DataLoader(self.train_set_tgt, batch_size=self.config.batch_size,
                                shuffle=True, pin_memory=True, num_workers=self.config.num_workers, drop_last=True),
            }, "max_size_cycle")

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
        transform = A.Compose([
            # A.Resize(self.config.img_size, self.config.img_size),
            # A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # source dataset
        self.train_set_src = NexmonDataset(root=self.config.root, fold=self.config.fold_no, train=True, transform=transform, channels=self.config.channels)
        self.val_set_src = NexmonDataset(root=self.config.root, fold=self.config.fold_no, train=False, transform=transform, channels=self.config.channels)

        if self.config.model_name != "timm":
            assert self.config.root_tgt is not None

            # target dataset
            if self.config.root_tgt_train is None:
                self.train_set_tgt = NexmonDataset(root=self.config.root_tgt, fold=self.config.fold_no, train=True, transform=transform, classes=self.train_set_src.classes, channels=self.config.channels)
                self.val_set_tgt = self.train_set_tgt
            else:
                # Zero-shot evaluation setting
                self.train_set_tgt = NexmonDataset(root=self.config.root_tgt_train, fold=self.config.fold_no, train=True, transform=transform, channels=self.config.channels, zs=True)
                self.val_set_tgt = NexmonDataset(root=self.config.root_tgt, fold=self.config.fold_no, train=True, transform=transform, channels=self.config.channels, zs=True)

        self.test_sets = get_all_nexmon_datasets(self.config.root, self.config.fold_no, transform, classes=self.train_set_src.classes, channels=self.config.channels)

    def init_params(self):
        self.num_classes = len(self.train_set_src.classes)
        self.num_test_sets = len(self.test_sets)


class NexmonDataset(datasets.VisionDataset):
    def __init__(self, root, fold, train, transform=None, classes=None, channels=3, zs=False, single_per_class=False):
        super().__init__(root, transform=transform)

        self.fold = fold
        self.train = train
        self.classes = classes
        self.channels = channels
        self.zs = zs

        self.single_per_class = single_per_class
        self.data = self.parse_data_file()

    def __getitem__(self, index):
        path, target = self.data[index]
        img = np.uint8(np.load(path).T * 255)
        # print(img.shape)
        # img = np.uint8(np.moveaxis(np.load(path), 0, -1) * 255)

        if img.shape == (256, 256, 4):
            img = [
                cv2.resize(img[:,:,0], (64, 256)),
                cv2.resize(img[:,:,1], (64, 256)),
                cv2.resize(img[:,:,2], (64, 256)),
                cv2.resize(img[:,:,3], (64, 256)),
            ]
            img = np.concatenate(img, axis=1)
            img = np.stack([
                img for _ in range(self.channels)
            ], axis=-1)

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = np.moveaxis(img, -1, 0).astype(np.float16)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        assert target >= 0

        if self.channels == 1:
            img = img[0:1,:,:]

        return img, target

    def __len__(self):
        return len(self.data)

    def parse_data_file(self):
        X = glob(os.path.join(self.root, "*", "*.npy"))
        y = [filename.split("/")[-1].replace(".npy", "").replace(".pcap", "").split("_")[-1][:-2] for filename in X]

        if not self.zs:
            X = [filename for filename in X if "noActivity" not in filename]
            y = [filename for filename in y if "noActivity" not in filename]

        if self.classes is not None:
            X, y = filter_data_by_classes(X, y, self.classes)
        else:
            self.classes = np.unique(y)

        # self.classes = ['circle', 'push', 'sitdown', 'swipe', 'upNdown', 'zigzag']
        # X = list(filter(lambda x: x.split("/")[-1].replace(".npy", "").split("_")[-1][:-2] in self.classes, X))

        class_map = dict([(x[1], x[0])for x in enumerate(self.classes)])
        y = [class_map[idx] for idx in y]

        X = np.array(X)
        y = np.array(y)

        if self.fold is None:
            assert len(X) != 0
            return list(zip(X, y))
        else:
            skf = StratifiedKFold(n_splits=10)
            train_index, test_index = list(skf.split(X, y))[self.fold]

            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]

            if self.train:
                assert len(X_train) != 0
                return list(zip(X_train, y_train))
            else:
                assert len(X_val) != 0
                if self.single_per_class:
                    X_val, y_val = filter_single_per_class(X_val, y_val, list(range(len(self.classes))))
                return list(zip(X_val, y_val))


def filter_data_by_classes(X, y, classes):
    filtered_X = []
    filtered_y = []
    for filename, label in zip(X, y):
        if label in classes:
            filtered_X.append(filename)
            filtered_y.append(label)
    return filtered_X, filtered_y


def filter_single_per_class(X, y, classes):
    new_X = []
    new_y = []
    for x_i, y_i in zip(X, y):
        if len(classes) == 0:
            break
        if y_i in classes:
            new_X.append(x_i)
            new_y.append(y_i)
            classes.remove(y_i)
        else:
            continue
    return new_X, new_y