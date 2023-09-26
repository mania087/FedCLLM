import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler
from utils import train, test, predict_step
from PIL import Image
from torchvision import transforms

class Client():
    def __init__(self, client_config:dict):
        # client config as dict to make configuration dynamic
        self.id = client_config["id"]
        self.config = client_config
        self.__model = None

        # as a marker
        self.is_malicious = client_config["is_malicious"]
        
        # check if CUDA is available
        if not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        # if we use validation
        if self.config["val_size"] > 0.0:
            num_train = len(self.config["train_data"])
            indices = list(range(num_train))
            np.random.shuffle(indices)
            split = int(np.floor(self.config["val_size"] * num_train))
            train_idx, valid_idx = indices[split:], indices[:split]

            # define samplers for obtaining training and validation batches
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            # prepare data loaders (combine dataset and sampler)
            self.train_loader = torch.utils.data.DataLoader(self.config["train_data"], 
                                                            batch_size=self.config["batch_size"],
                                                            sampler=train_sampler)
            self.valid_loader = torch.utils.data.DataLoader(self.config["train_data"],
                                                            batch_size=self.config["batch_size"],
                                                            sampler=valid_sampler) 
            
            
            # save raw train data
            self.raw_train_data = [self.config["train_data"][x] for x in train_idx]
        else:
            self.train_loader = torch.utils.data.DataLoader(self.config["train_data"], 
                                                            batch_size=self.config["batch_size"])
            self.valid_loader = None
            
            # save raw train data
            self.raw_train_data = self.config["train_data"]

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.train_loader.sampler)

    def train(self, algorithm):
        results= {}
        if algorithm == "FedAvg":
            # FedAvg algorithm
            results = train(net=self.model, 
                            trainloader= self.train_loader, 
                            epochs= self.config["local_epoch"],
                            device= self.device, 
                            valloader= self.valid_loader)
        else:
            # other algorithm
            pass
        print(f"Train result client {self.id}: {results}")
    
    def test(self):
        loss,acc = test(net = self.model, 
                        testloader = self.valid_loader,
                        device=self.device)
        print(f"Test result client {self.id}: {loss, acc}")