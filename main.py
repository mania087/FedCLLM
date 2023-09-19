import numpy as np
import torch
import random
import copy
from dataloader import create_datasets
from client import Client
from model import CnnModel
from utils import test
## set seed
np.random.seed(seed=42)
torch.manual_seed(0)

## Experiment Parameters
algorithm = "FedAvg"
num_clients = 100
num_classes = 10
local_epoch = 1
val_size = 0.0
batch_size = 32
fl_rounds = 100
C = 0.2 # clients availability
iid = True

## load dataset for clients and evaluator
local_datasets, test_dataset= create_datasets(data_path='data', 
                                              dataset_name='CIFAR10', 
                                              num_clients=num_clients, 
                                              num_shards=200, 
                                              iid=iid, 
                                              transform=None, 
                                              print_count=True)

# create evaluator test dataset
evaluator_validation = torch.utils.data.DataLoader(test_dataset,batch_size=1) 

## do federated learning between honest clients and malicious clients
# create honest clients 
clients = []
for index, dataset in enumerate(local_datasets):
    clients.append(
        Client(
            {"id": index,
             "train_data": dataset,
             "val_size": val_size,
             "batch_size": batch_size,
             "is_malicious": False,
             "local_epoch": local_epoch
            }
        )
    )

# shuffle the client
random.shuffle(clients)

print("Clients created...")
print(f"Number of  clients: {num_clients}")

## start Federated Learning cycle:

### FedAvg:
if algorithm=="FedAvg":
    ## Set algorithm specific requirements

    ## set server model 
    server_model = CnnModel(input_size=(3,28,28), num_classes=num_classes)

    for train_round in range(fl_rounds):
        print(f'Round {train_round}...')
        ## number of clients to pick
        num_clients_to_pick = int(C * len(clients))
        ## pick clients
        round_selected_clients = np.random.choice(len(clients), num_clients_to_pick, replace=False)
        selected_clients = [clients[client_index] for client_index in round_selected_clients]
        print(f'Training with {len(selected_clients)} clients...')

        for train_clients in selected_clients:
            ## get updated model from server
            train_clients.model = copy.deepcopy(server_model)
            ## train clients
            train_clients.train(algorithm=algorithm)
        
        ## Update server model
        total_data_points = sum([len(train_clients)for train_clients in selected_clients])
        fed_avg_freqs = [len(train_clients)/ total_data_points for train_clients in selected_clients]

        global_w = server_model.state_dict()
        for net_id, train_clients in enumerate(selected_clients):
            net_para = train_clients.model.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * fed_avg_freqs[net_id]
        
        ## global model load new weights
        server_model.load_state_dict(global_w)

        ## test model
        print(test(server_model, evaluator_validation))

### Other algorithm
else:
    pass