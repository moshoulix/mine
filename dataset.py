"""
MNIST
"""

from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from torchvision import datasets, transforms
import os


def get_dataset(split):
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    data_root = './dataset'
    if split == 'train':
        return datasets.MNIST(root=data_root, train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    elif split == 'test':
        return datasets.MNIST(root=data_root, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))


def mnist_dataloader(sp, batch_size, train_num=None, shuffle=True, drop_last=False):
    if train_num is None:
        return DataLoader(dataset=get_dataset(sp), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    else:
        sampler = SubsetRandomSampler(range(train_num))
        return DataLoader(dataset=get_dataset(sp), batch_size=batch_size, shuffle=shuffle,
                          drop_last=drop_last, sampler=sampler)


if __name__ == '__main__':
    get_dataset('train')
    get_dataset('test')
