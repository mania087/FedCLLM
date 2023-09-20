
import torch
import numpy as np
import random
import torchvision
import math
from torch.utils.data import Dataset

class To3dFrom1D(object):
    """Convert 1d to 3d."""

    def __call__(self, image):

        # expand dimension
        image = torch.cat((image, image, image), dim=0)
        return image
    
class CustomDataset(Dataset):
    def __init__(self, tensors, transforms=None):
        self.tensors =tensors
        self.transforms = transforms

    def __len__(self):
        return self.tensors[0].size(0)
    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        if self.transforms:
            x = self.transforms(x.numpy().astype(np.uint8))
        return x,y
    
def create_datasets(data_path, dataset_name, num_clients=100, num_shards=200, iid=True, transform=None, print_count=None):

    # check dataset
    if dataset_name == "CIFAR10":
        # check if transform is defined:
        if transform:
            preprocess = transform
        else:
            preprocess = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

        training_dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            download=True,
            transform = preprocess
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            download=True,
            transform = preprocess
        )

    elif dataset_name == "MNIST":
        # check if transform is defined:
        if transform:
            preprocess = transform
        else:
            preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 To3dFrom1D()]
            )

        training_dataset = torchvision.datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform = preprocess
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_path,
            train=False,
            download=True,
            transform = preprocess
        )
    if training_dataset.data.ndim ==3: # make it batch (NxWxH => NxWxHx1)
        training_dataset.data.unsqueeze_(3)

    # unique labels
    num_categories = np.unique(training_dataset.targets).shape[0]

    # For MNIST
    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    if "list" not in str(type(training_dataset.targets)):
        training_dataset.targets = training_dataset.targets.tolist()

    if iid:
        # shuffle data
        shuffle = torch.randperm(len(training_dataset))
        training_inputs = training_dataset.data[shuffle]
        training_labels = torch.tensor(training_dataset.targets)[shuffle]

        # partition into clients
        split_size = len(training_dataset) // num_clients
        # partition information
        stack_of_label = torch.stack(list(torch.split(training_labels, split_size)), dim=0)
        count = torch.nn.functional.one_hot (stack_of_label).sum(dim = 1)
        if print_count:
            print(count)

        split_datasets = list(zip(
            torch.split(torch.Tensor(training_inputs), split_size),
            torch.split(training_labels, split_size)
            ))

        local_datasets = []
        for i, local_dataset in enumerate(split_datasets):
            local_datasets.append(
                CustomDataset(local_dataset,transforms = preprocess)
                )
    else:
        # Non-IID split
        # first, sort data by labels
        sorting_idx = torch.argsort(torch.Tensor(training_dataset.targets))
        training_inputs = training_dataset.data[sorting_idx]
        training_labels = torch.tensor(training_dataset.targets)[sorting_idx]

        # second partition data into shards
        shard_size = len(training_dataset)//num_shards
        shard_inputs = torch.split(torch.Tensor(training_inputs), shard_size)
        shard_labels = torch.split(training_labels, shard_size)

        # sort the list to assign samples to each client
        # from at least 2 classes
        shard_inputs_sorted, shard_labels_sorted = [], []
        for i in range(num_shards // num_categories):
            for j in range(0, ((num_shards // num_categories) * num_categories), (num_shards // num_categories)):
                shard_inputs_sorted.append(shard_inputs[i+j])
                shard_labels_sorted.append(shard_labels[i+j])

        # partition information
        shards_per_clients = num_shards // num_clients
        test = [
            torch.cat(shard_labels_sorted[i:i+shards_per_clients])
            for i in range(0, len(shard_inputs_sorted), shards_per_clients)
        ] 
        stack_of_label = torch.stack(test,dim=0)
        count = torch.nn.functional.one_hot (stack_of_label).sum(dim = 1)
        if print_count:
            print(count)

        local_datasets = []
        just_count = 0
        for i in range(0, len(shard_inputs_sorted), shards_per_clients):
            local_datasets.append(
                CustomDataset(
                    (
                        torch.cat(shard_inputs_sorted[i:i+shards_per_clients]),
                        torch.cat(shard_labels_sorted[i:i+shards_per_clients]),
                    ),
                    transforms = preprocess
                )
            )
            just_count += 1

    return local_datasets, test_dataset

