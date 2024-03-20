
import torch
import numpy as np
import random
import torchvision
import math
import scipy.io
import glob

from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

class To3dFrom1D(object):
    """Convert 1d to 3d."""

    def __call__(self, image):

        # expand dimension
        image = torch.cat((image, image, image), dim=0)
        return image
    
class CustomDataset(Dataset):
    def __init__(self, data, targets, transforms=None, from_list_link=False, return_raw=False):
        self.data =data
        self.targets = targets
        self.transforms = transforms
        
        self.from_list_link = from_list_link
        self.return_raw = return_raw
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.from_list_link:
            img = Image.open(self.data[index])
            x = img.copy()
            img.close()
        else:
            x = self.data[index]
            
        y = self.targets[index]
        
        if self.return_raw:
            return x, y
        else:
            if self.transforms:
                x = self.transforms(x)
            return x,y

def oxford_flower_make_data(original_image, original_label, idx):
    data = []
    list_labels = []
    for i in idx:
        # to handle memory / open to many files
        data.append(original_image[i])
        list_labels.append(original_label[i])
    return data, list_labels

def create_datasets(data_path, 
                    dataset_name, 
                    num_clients=100, 
                    separate_validation_data=True,
                    val_size=0.2,
                    num_shards=200, iid=True, transform=None, print_count=None):
    
    val_dataset = None
    
    # check dataset
    if dataset_name == "CIFAR10":
        # check if transform is defined:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        from_list_link = False
        if transform:
            preprocess = transform
        else:
            preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224), 
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ])

        training_dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            download=True,
            transform = None
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            download=True,
            transform = preprocess
        )

    elif dataset_name == "MNIST":
        # check if transform is defined:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        from_list_link = False
        if transform:
            preprocess = transform
        else:
            preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(224), 
                 torchvision.transforms.ToTensor(),
                 To3dFrom1D(),
                 torchvision.transforms.Normalize(mean, std),
                ]
            )

        training_dataset = torchvision.datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform = None
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_path,
            train=False,
            download=False,
            transform = preprocess
        )
        
    elif dataset_name == "FMNIST":
        # check if transform is defined:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        from_list_link = False
        if transform:
            preprocess = transform
        else:
            preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(224), 
                 torchvision.transforms.ToTensor(),
                 To3dFrom1D(),
                 torchvision.transforms.Normalize(mean, std),
                ]
            )

        training_dataset = torchvision.datasets.FashionMNIST(
            root=data_path,
            train=True,
            download=True,
            transform = None
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_path,
            train=False,
            download=True,
            transform = preprocess
        )
    # oxford 102 dataset
    elif dataset_name == "Oxford102":
        std = (0.229, 0.224, 0.225)
        mean = (0.485, 0.456, 0.406)
        from_list_link = True
        if transform:
            preprocess = transform
        else:
            preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(224), 
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(mean, std),
                ]
            )
        
        #load dataset
        image_paths = sorted(glob.glob(data_path+'/102flowers/jpg/' + "*.jpg"))
        labels_id = scipy.io.loadmat(data_path+'/102flowers/imagelabels.mat')['labels'][0]
        # make it start from 0
        labels_id -= 1

        # Read .mat file containing training, testing, and validation sets.
        setid = scipy.io.loadmat(data_path+'/102flowers/setid.mat')

        # The .mat file is 1-indexed, so we subtract one 
        idx_train = setid['trnid'][0] - 1
        idx_test = setid['tstid'][0] - 1
        idx_valid = setid['valid'][0] - 1
        print(len(idx_train), len(idx_test), len(idx_valid))

        # shuffle
        idx_train = idx_train[np.random.permutation(len(idx_train))]
        idx_test = idx_test[np.random.permutation(len(idx_test))]
        idx_valid = idx_valid[np.random.permutation(len(idx_valid))]
        
        train_data, train_labels = oxford_flower_make_data(image_paths, labels_id, idx_train)   
        test_data, test_labels = oxford_flower_make_data(image_paths, labels_id, idx_test)
        val_data, val_labels = oxford_flower_make_data(image_paths, labels_id, idx_valid)
        
        training_dataset = CustomDataset(
            train_data,
            train_labels,
            transforms=preprocess,
            from_list_link=from_list_link
        )
        print(len(training_dataset))
        val_dataset = CustomDataset(
            val_data,
            val_labels,
            transforms=preprocess,
            from_list_link=from_list_link
        )
        print(len(val_dataset))
        test_dataset = CustomDataset(
            test_data,
            test_labels,
            transforms=preprocess,
            from_list_link=from_list_link
        )
        print(len(test_dataset))
            
    # tiny imagenet dataset
    elif dataset_name == "TinyImagenet":
        std = (0.229, 0.224, 0.225)
        mean = (0.485, 0.456, 0.406)
        from_list_link = True
        if transform:
            preprocess = transform
        else:
            preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(224), 
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(mean, std),
                ]
            )
            
        #load dataset
        training_dataset = torchvision.datasets.ImageFolder(
            root=data_path+'/tiny-imagenet-200/train/',
            transform = preprocess
        )
        # create .data and .targets for the dataset
        training_data = [x[0] for x in training_dataset.imgs]
        training_targets = [x[1] for x in training_dataset.imgs]
        
        training_dataset = CustomDataset(
            training_data,
            training_targets,
            transforms=preprocess,
            from_list_link=from_list_link
        )
        val_dataset = torchvision.datasets.ImageFolder(
            root=data_path+'/tiny-imagenet-200/val/',
            transform = preprocess
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=data_path+'/tiny-imagenet-200/test/',
            transform = preprocess
        )
        
    # food 101 dataset
    elif dataset_name == "Food101":
        std = (0.229, 0.224, 0.225)
        mean = (0.485, 0.456, 0.406)
        from_list_link = True
        if transform:
            preprocess = transform
        else:
            preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(224), 
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(mean, std),
                ]
            )
        # read labels file
        images_path = data_path+ '/food-101/images/'
        label_converter = {}
        with open(data_path+'/food-101/meta/classes.txt') as file:
            for line in file:
                label_name = line.rstrip()
                label_converter[label_name] = len(label_converter)

        # read training data_list
        train_images = []
        train_labels = []
        with open(data_path+'/food-101/meta/train.txt') as file:
            for line in file:
                dir_path=line.rstrip()
                label_name = dir_path.split('/')[0]
                train_images.append(images_path+dir_path+'.jpg')
                train_labels.append(label_converter[label_name])
                
        # read test data_list
        test_images = []
        test_labels = []
        with open(data_path+'/food-101/meta/test.txt') as file:
            for line in file:
                dir_path=line.rstrip()
                label_name = dir_path.split('/')[0]
                test_images.append(images_path+dir_path+'.jpg')
                test_labels.append(label_converter[label_name])
        
        training_dataset = CustomDataset(
            train_images,
            train_labels,
            transforms=preprocess,
            from_list_link=from_list_link
        )
        
        test_dataset = CustomDataset(
            test_images,
            test_labels,
            transforms=preprocess,
            from_list_link=from_list_link
        )
    
    print()
    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    
    if "ndarray" not in str(type(training_dataset.targets)):
        training_dataset.targets = np.asarray(training_dataset.targets,dtype=np.int64)  
        
    # create validation dataset
    if separate_validation_data and dataset_name !='TinyImagenet' and dataset_name !='Oxford102':
        # use train test split to split indexes
        list_train_indexes = list(range(len(training_dataset)))
        train_indexes, _, val_indexes, _ = train_test_split(list_train_indexes, 
                                                            training_dataset.targets, 
                                                            test_size=val_size, 
                                                            random_state=42, 
                                                            stratify=training_dataset.targets)
        print(type(training_dataset.targets))
        # using indexes to built the datasets
        new_train_dataset = [training_dataset.data[train_indexes], training_dataset.targets[train_indexes]]
        validation_dataset = [training_dataset.data[val_indexes], training_dataset.targets[val_indexes]]

        training_dataset = CustomDataset(new_train_dataset[0], new_train_dataset[1], transforms = preprocess, from_list_link=from_list_link)
        val_dataset = CustomDataset(validation_dataset[0], validation_dataset[1], transforms = preprocess, from_list_link=from_list_link)
    
    if training_dataset.data.ndim ==3: # make it batch (NxWxH => NxWxHx1)
        training_dataset.data= np.expand_dims(training_dataset.data, axis=3)
        
    # unique labels
    num_categories = np.unique(training_dataset.targets).shape[0]
    
    # TODO: change to numpy based since there is some dataset that does not load into image directly
    # due to size limitation
    if iid:
        # shuffle data
        shuffle = np.random.permutation(len(training_dataset))
        training_inputs = training_dataset.data[shuffle]
        training_labels = training_dataset.targets[shuffle]

        # partition information
        stack_of_label = np.stack(
            list(np.array_split(training_labels, num_clients))
            ,axis=0)
        # for counting label
        count = torch.nn.functional.one_hot(torch.from_numpy(stack_of_label)).sum(dim = 1)
        if print_count:
            print(count)
        
        split_datasets = list(zip(
            np.array_split(training_inputs, num_clients),
            np.array_split(training_labels, num_clients)
        ))

        local_datasets = []
        for i, local_sharp in enumerate(split_datasets):
            local_datasets.append(
                CustomDataset(local_sharp[0],
                              local_sharp[1],
                              transforms = preprocess,
                              from_list_link=from_list_link)
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
            
    if separate_validation_data:
        return local_datasets, test_dataset, val_dataset
    else:
        return local_datasets, test_dataset

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_path = "../../dataset"
    dataset_name = "Oxford102"
    
    local_datasets, test_dataset, val_dataset= create_datasets(data_path, 
                                                               dataset_name, 
                                                               num_clients=10, 
                                                               num_shards=10, iid=True, transform=None, print_count=True)
    print(len(val_dataset))
    sample_index = 1
    ex_dataset = local_datasets[0]
    # output raw
    ex_dataset.return_raw = True
    sample_image = ex_dataset[sample_index][0]
    sample_label = ex_dataset[sample_index][1]
    # Normalize data
    fig, ax = plt.subplots()
    
    print(f"Label {sample_label}")
    ax.imshow(sample_image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.show()
