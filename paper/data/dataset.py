import os
import subprocess
import pickle
import numpy as np
import torch

from torchvision import datasets, transforms


class RandomSampler(torch.utils.data.Sampler):

    def __init__(self, data, batch_size):
        self.length = len(data)
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield np.random.randint(self.length, size=self.batch_size)


class SimpleLLMDataset(torch.utils.data.Dataset):

    def __init__(self, data, context):
        self.data = data
        self.context = context

    def __getitem__(self, index):
        return torch.tensor(self.data[index  :index+self.context  ].astype(np.int64)), \
               torch.tensor(self.data[index+1:index+self.context+1].astype(np.int64))

    def __len__(self):
        return len(self.data) - self.context - 1


def getDataset(dataset, context=None):

    if dataset == "cifar10":

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

        input_dim = (3,32,32)
        output_dim = 10

        return trainset, testset, input_dim, output_dim

    elif dataset in ["shakespeare", "tinystories", "openwebtext"]:

        if not os.path.exists(f"data/{dataset}/train.bin"):
            subprocess.call(['python', f"data/{dataset}.py"])

        trainset = SimpleLLMDataset(np.memmap(f"data/{dataset}/train.bin", dtype=np.uint16, mode='r'), context)
        testset  = SimpleLLMDataset(np.memmap(f"data/{dataset}/val.bin",   dtype=np.uint16, mode='r'), context)

        vocab_size = 65 if dataset == "shakespeare" else 50257

        return trainset, testset, vocab_size, None

    else: raise ValueError(f"Unknown dataset: {dataset}")


def getIterator(dataset, batch_size, context=None):

    trainset, testset, input_dim, output_dim = getDataset(dataset, context)

    train_sampler = RandomSampler(trainset, batch_size)
    test_sampler  = RandomSampler(testset,  batch_size)

    train_loader = torch.utils.data.DataLoader( trainset, num_workers=16, pin_memory=True, batch_sampler=train_sampler)
    test_loader  = torch.utils.data.DataLoader( testset,  num_workers=16, pin_memory=True, batch_sampler=test_sampler)

    train_iterator = iter(train_loader)
    test_iterator  = iter(test_loader)

    getBatch = lambda train: next(train_iterator if train else test_iterator)

    return getBatch, input_dim, output_dim
