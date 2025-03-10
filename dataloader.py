
import torch
import numpy as np
import random
import torchvision
import math
import scipy.io
import glob
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_RESIZE = (224, 224)

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
            x = img.copy().convert('RGB')
            img.close()
        else:
            # from numpy array
            x = self.data[index]
            # if last array is 1d, convert to 3d
            if x.shape[-1] == 1:
                x = np.concatenate((x, x, x), axis=-1)
            x = Image.fromarray(np.uint8(x))
            
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
                    num_shards=200, iid=True, non_iid_mode='shard', dir_alpha=10, transform=None, print_count=None):
    
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
                torchvision.transforms.Resize(IMG_RESIZE), 
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ])
        
        test_preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize(IMG_RESIZE), 
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
            transform = test_preprocess
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
                [torchvision.transforms.Resize(IMG_RESIZE), 
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(mean, std),
                ]
            )
        
        test_preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(IMG_RESIZE), 
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(mean, std),
                ]
            )

        training_dataset = torchvision.datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform = None
        )
        # for torch native dataset, there is no 1d-to 3d so create manually
        test_dataset = torchvision.datasets.MNIST(
            root=data_path,
            train=False,
            download=False,
            transform = test_preprocess
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
                [torchvision.transforms.Resize(IMG_RESIZE), 
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(mean, std),
                ]
            )
        
        test_preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(IMG_RESIZE), 
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
        val_dataset = CustomDataset(
            val_data,
            val_labels,
            transforms=test_preprocess,
            from_list_link=from_list_link
        )

        test_dataset = CustomDataset(
            test_data,
            test_labels,
            transforms=test_preprocess,
            from_list_link=from_list_link
        )
            
    # tiny imagenet dataset
    elif dataset_name == "TinyImagenet":
        std = (0.229, 0.224, 0.225)
        mean = (0.485, 0.456, 0.406)
        from_list_link = True
        if transform:
            preprocess = transform
        else:
            preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(IMG_RESIZE), 
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(mean, std),
                ]
            )
        
        test_preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(IMG_RESIZE), 
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(mean, std),
                ]
            )
            
        #load dataset
        training_dataset = torchvision.datasets.ImageFolder(
            root=data_path+'/tiny-imagenet-200/train/',
            transform = preprocess
        )
        
        val_dataset = torchvision.datasets.ImageFolder(
            root=data_path+'/tiny-imagenet-200/val/',
            transform = test_preprocess
        )
        
        test_dataset = torchvision.datasets.ImageFolder(
            root=data_path+'/tiny-imagenet-200/test/',
            transform = test_preprocess
        )
        
        # NOTE: for temporary, take only 100 classes of imagenet for category compability
        all_classes = list(range(0,200))
        random.shuffle(all_classes)
        
        allowed_classes = all_classes[:100]
        dictionary_remap = {class_name:index for index,class_name in enumerate(allowed_classes)}    
                
        # create .data and .targets for the dataset
        training_data = [x[0] for x in training_dataset.imgs]
        training_targets = [x[1] for x in training_dataset.imgs]
        
        val_data = [x[0] for x in val_dataset.imgs]
        val_targets = [x[1] for x in val_dataset.imgs]
        
        test_data = [x[0] for x in test_dataset.imgs]
        test_targets = [x[1] for x in test_dataset.imgs]
        
        # allow data within the allowed classes
        ## search indexes first
        training_indexes = [i for i, x in enumerate(training_targets) if x in allowed_classes]
        val_indexes = [i for i, x in enumerate(val_targets) if x in allowed_classes]
        test_indexes = [i for i, x in enumerate(test_targets) if x in allowed_classes]
        
        ## get the data based on indexes and rename
        training_data = [training_data[i] for i in training_indexes]
        training_targets = [dictionary_remap[training_targets[i]] for i in training_indexes]
        
        val_data = [val_data[i] for i in val_indexes]   
        val_targets = [dictionary_remap[val_targets[i]] for i in val_indexes]
        
        test_data = [test_data[i] for i in test_indexes]
        test_targets = [dictionary_remap[test_targets[i]] for i in test_indexes]
        
        training_dataset = CustomDataset(
            training_data,
            training_targets,
            transforms=preprocess,
            from_list_link=from_list_link
        )
        
        val_dataset = CustomDataset(
            val_data,
            val_targets,
            transforms=test_preprocess,
            from_list_link=from_list_link
        )
        
        test_dataset = CustomDataset(
            test_data,
            test_targets,
            transforms=test_preprocess,
            from_list_link=from_list_link
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
                [torchvision.transforms.Resize(IMG_RESIZE), 
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(mean, std),
                ]
            )
        
        test_preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(IMG_RESIZE), 
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
            transforms=test_preprocess,
            from_list_link=from_list_link
        )
    
    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    
    if "ndarray" not in str(type(training_dataset.targets)):
        training_dataset.targets = np.asarray(training_dataset.targets,dtype=np.int64)  
    
    if training_dataset.data.ndim ==3: # make it batch (NxWxH => NxWxHx1)
        training_dataset.data= np.expand_dims(training_dataset.data, axis=3)
        test_dataset.data = np.expand_dims(test_dataset.data, axis=3)
    
    # NOTE: until there is a better way
    # native pytorch dataset use img = Image.fromarray(img.numpy(), mode="L")
    # we can't use that since it's already on numpy
    test_dataset= CustomDataset(test_dataset.data, test_dataset.targets, transforms = test_preprocess, from_list_link=from_list_link)
        
    # create validation dataset
    if separate_validation_data and dataset_name !='TinyImagenet' and dataset_name !='Oxford102':
        # use train test split to split indexes
        list_train_indexes = list(range(len(training_dataset)))
        train_indexes, _, val_indexes, _ = train_test_split(list_train_indexes, 
                                                            training_dataset.targets, 
                                                            test_size=val_size, 
                                                            random_state=42, 
                                                            stratify=training_dataset.targets)
        # using indexes to built the datasets
        new_train_dataset = [training_dataset.data[train_indexes], training_dataset.targets[train_indexes]]
        validation_dataset = [training_dataset.data[val_indexes], training_dataset.targets[val_indexes]]

        training_dataset = CustomDataset(new_train_dataset[0], new_train_dataset[1], transforms = preprocess, from_list_link=from_list_link)
        val_dataset = CustomDataset(validation_dataset[0], validation_dataset[1], transforms = test_preprocess, from_list_link=from_list_link)
            
    # unique labels
    category = np.unique(training_dataset.targets)
    num_categories = np.unique(training_dataset.targets).shape[0]
    
    if iid:
        # shuffle data
        shuffle = np.random.permutation(len(training_dataset))
        training_inputs = training_dataset.data[shuffle]
        training_labels = training_dataset.targets[shuffle]

        # partition information
        stack_of_label = np.array_split(training_labels, num_clients)
        # get the count of each label
        count = []
        for split in stack_of_label:
            # for counting label
            one_hot_count = [len(split[split == i]) for i in category]
            if print_count:
                print(one_hot_count)

        
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
    elif not iid and non_iid_mode == 'dirichlet':
        # Non-IID split
        # first, sort data by labels
        sorting_idx = np.argsort(training_dataset.targets)
        training_inputs = training_dataset.data[sorting_idx]
        training_labels = training_dataset.targets[sorting_idx]

        # get the list of index related to each label
        partition_info = []
        clients_shard= [[] for _ in range(num_clients)]
        for label in category:
            indexes_location = np.where(training_labels == label)[0]
            # shuffle the indexes
            np.random.shuffle(indexes_location)
            # get the dirilecht partition of number of data
            dir_partition = np.random.multinomial(len(indexes_location), np.random.dirichlet(np.ones(num_clients) * dir_alpha)) 
            partition_info.append(dir_partition)
            splitted_index = np.array_split(indexes_location, np.cumsum(dir_partition))[:-1]

            # append the data into clients_shard
            for i, split in enumerate(splitted_index):
                clients_shard[i].extend(split)
        
        partition_info = np.transpose(np.vstack(partition_info))
        if print_count:
            print(partition_info)
        
        # assign the shard to the clients
        local_datasets = []
        for i, shard in enumerate(clients_shard):
            local_datasets.append(
                CustomDataset(
                    training_inputs[shard],
                    training_labels[shard],
                    transforms = preprocess,
                    from_list_link=from_list_link
                )
            )
            
    elif not iid and non_iid_mode == 'shard':
        # Non-IID split
        # first, sort data by labels
        sorting_idx = np.argsort(training_dataset.targets)
        training_inputs = training_dataset.data[sorting_idx]
        training_labels = training_dataset.targets[sorting_idx]

        # second partition data into shards
        shard_size = len(training_dataset)//num_shards
        shard_inputs = np.split(training_inputs, shard_size)
        shard_labels = np.split(training_labels, shard_size)

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
            np.concatenate(shard_labels_sorted[i:i+shards_per_clients])
            for i in range(0, len(shard_inputs_sorted), shards_per_clients)
        ] 

        stack_of_label = np.stack(test,axis=0)
        count = torch.nn.functional.one_hot(torch.from_numpy(stack_of_label)).sum(dim = 1)
        if print_count:
            print(count)

        local_datasets = []
        just_count = 0
        for i in range(0, len(shard_inputs_sorted), shards_per_clients):
            local_datasets.append(
                CustomDataset(
                    np.concatenate(shard_inputs_sorted[i:i+shards_per_clients]),
                    np.concatenate(shard_labels_sorted[i:i+shards_per_clients]),
                    transforms = preprocess,
                    from_list_link=from_list_link
                )
            )
            just_count += 1

    if separate_validation_data:
        return local_datasets, test_dataset, val_dataset, category
    else:
        return local_datasets, test_dataset, category