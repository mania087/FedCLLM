import numpy as np
import torch
import torch.nn.functional as F
import time
from torch.utils.data.sampler import SubsetRandomSampler
from utils import train, test

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
        
        # for fedcs
        self.time_update = 0.0
        self.time_upload = 0.0

        self.train_dataset = self.config["train_data"]
        
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
                                                            sampler=train_sampler,
                                                            drop_last=True)
            self.valid_loader = torch.utils.data.DataLoader(self.config["train_data"],
                                                            batch_size=self.config["batch_size"],
                                                            sampler=valid_sampler) 
            
        else:
            self.train_loader = torch.utils.data.DataLoader(self.config["train_data"], 
                                                            batch_size=self.config["batch_size"])
            self.valid_loader = None
            

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.train_loader.sampler)
    
    def train(self, algorithm, lr, opt, verbose=False):
        results= {}
        start_time = time.time()
        if algorithm == "FedAvg":
            # FedAvg algorithm
            results = train(net=self.model, 
                            trainloader= self.train_loader, 
                            epochs= self.config["local_epoch"],
                            lr=lr,
                            opt=opt,
                            device= self.device, 
                            valloader= self.valid_loader,
                            verbose=verbose)
        else:
            # other algorithm
            pass
        # update time for fedcs
        self.time_update = time.time() - start_time
        
        if verbose:
            print(f"Train result client {self.id}: {results}")
    
    def test(self):
        loss,acc = test(net = self.model, 
                        testloader = self.valid_loader,
                        device=self.device)
        print(f"Test result client {self.id}: {loss, acc}")
        