from omegaconf import DictConfig
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Grayscale
from skimage import feature

import hydra
import os
import skimage
import skimage.io
import skimage.transform
import torchvision
import torch
import tqdm

import pickle as pkl
import numpy as np

# from nist import NIST


def compose_image(digit, background):
  """Difference-blend a digit and a random patch from a background image."""
  w, h, _ = background.shape
  dw, dh, _ = digit.shape
  x = np.random.randint(0, w - dw)
  y = np.random.randint(0, h - dh)

  bg = background[x:x+dw, y:y+dh]
  return np.abs(bg - digit).astype(np.uint8)


def preprocess(x, img_size, rgb=True):
  """Binarize MNIST digit and convert to RGB."""
  if type(x) == torch.Tensor:
    x = x.numpy()

  if img_size == 128:
    # If dataset is NIST
    gray_scaler = Grayscale()
    x = np.array(gray_scaler(default_loader(x)))
  x = (x > 0).astype(np.float32)
  d = x.reshape([img_size, img_size, 1]) * 255
  if rgb is True:
    return np.concatenate([d, d, d], 2)
  else:
    return d


def create_c(X, img_size, bst_path):
  """
  Give an array of MNIST digits, blend random background patches to
  build the MNIST-M dataset as described in
  http://jmlr.org/papers/volume17/15-239/15-239.pdf
  """
  rand = np.random.RandomState(42)
  print('Loading BSR training images')
  train_files = []
  for name in os.listdir(bst_path):
    train_files.append(os.path.join(bst_path, name))

  background_data = []
  for name in train_files:
    if ".jpg" in name:
      bg_img = skimage.io.imread(name)
      background_data.append(bg_img)

  X_ = np.zeros([X.shape[0], img_size, img_size, 3], np.uint8)
  for i in tqdm.tqdm(range(X.shape[0])):

    bg_img = rand.choice(background_data)

    d = preprocess(X[i], img_size)
    d = compose_image(d, bg_img)
    X_[i] = d
  return X_


def create_e(X, img_size, bst_path):
  X_ = np.zeros([X.shape[0], img_size, img_size, 3], np.uint8)
  for i in tqdm.tqdm(range(X.shape[0])):
    d = preprocess(X[i], img_size, rgb=False)[:, :, 0]
    d = skimage.feature.canny(d)
    d = np.expand_dims(d, axis=-1)
    X_[i] = d * 255
  return X_


def create_n(X, img_size, bst_path):
  X_ = np.zeros([X.shape[0], img_size, img_size, 3], np.uint8)
  for i in tqdm.tqdm(range(X.shape[0])):
    d = preprocess(X[i], img_size, rgb=False)[:, :, 0]
    d = 255 - d
    d = np.expand_dims(d, axis=-1)
    X_[i] = d
  return X_


@hydra.main(config_path=".", config_name="config")
def create_color_domain(cfg: DictConfig) -> None:
  bst_path = cfg.bst_path
  data_path = cfg.data_path
  dataset = cfg.dataset
  domain = cfg.domain

  if domain == "C":
    create_function = create_c
  if domain == "E":
    create_function = create_e
  if domain == "N":
    create_function = create_n

  img_size = 28
  if dataset == "MNIST":
    train_dataset = torchvision.datasets.MNIST(
        root=os.path.join(data_path, "MNIST_G"), train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root=os.path.join(data_path, "MNIST_G"), train=False, download=True)
  if dataset == "FashionMNIST":
    train_dataset = torchvision.datasets.FashionMNIST(
        root=os.path.join(data_path, "FashionMNIST_G"), train=True, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(
        root=os.path.join(data_path, "FashionMNIST_G"), train=False, download=True)
  if dataset == "EMNIST":
    train_dataset = torchvision.datasets.EMNIST(
        root=os.path.join(data_path, "EMNIST_G"), train=True, split="letters", download=True)
    test_dataset = torchvision.datasets.EMNIST(
        root=os.path.join(data_path, "EMNIST_G"), train=False, split="letters", download=True)
  if dataset == "NIST":
    train_dataset = NIST(root=data_path, train=True)
    train_dataset.targets = torch.Tensor(train_dataset.targets)
    test_dataset = NIST(root=data_path, train=False)
    test_dataset.targets = torch.Tensor(train_dataset.targets)
    img_size = 128

  print('Building train set...')
  train_data = create_function(train_dataset.data, img_size, bst_path)
  train_target = train_dataset.targets.numpy()
  print('Building test set...')
  test_data = create_function(test_dataset.data, img_size, bst_path)
  test_target = test_dataset.targets.numpy()
  # Save dataset as pickle
  os.makedirs(os.path.join(data_path, f"{dataset}_{domain}"), exist_ok=True)
  with open(os.path.join(data_path, f"{dataset}_{domain}", f"data.pkl"), 'wb') as f:
    pkl.dump({
        "train": {"data": train_data, "targets": train_target},
        "test": {"data": test_data, "targets": test_target}
    }, f, pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
  create_color_domain()
