import numpy as np
import torch
import random
import copy
import openai 
from dataloader import create_datasets
from client import Client
from model import CnnModel
from utils import test, get_image_to_text_model, get_data_description, get_completion, get_api_key

## set key
openai.api_key  = get_api_key('backup/gpt_key.txt')

## set seed
np.random.seed(seed=42)
torch.manual_seed(0)

## Experiment Parameters
algorithm = "Proposed"
num_clients = 10
num_malicious_clients = 5
num_classes = 10
local_epoch = 3
val_size = 0.2
batch_size = 64
fl_rounds = 100
C = 1.0 # clients availability
iid = True

## Proposed method parameters
num_sample = 5
num_evaluator_sample = 5

## load dataset for clients and malicious clients
honest_client_numbers = num_clients - num_malicious_clients

## NOTE: somehow handling fraction using num clients and shards are hard, 
## There are some consideration to use dirilect
local_datasets, test_dataset= create_datasets(data_path='data', 
                                              dataset_name='MNIST', 
                                              num_clients=100, 
                                              num_shards=200, 
                                              iid=iid, 
                                              transform=None, 
                                              print_count=True)
if num_malicious_clients > 0:
    malicious_datasets, _= create_datasets(data_path='data', 
                                        dataset_name='CIFAR10', 
                                        num_clients=num_malicious_clients, 
                                        num_shards=200, 
                                        iid=iid, 
                                        transform=None, 
                                        print_count=True)

# pick honest client datasets
selected_indices = np.random.choice(len(local_datasets), honest_client_numbers, replace=False)
honest_clients_dataset = [local_datasets[x] for x in selected_indices]

# create evaluator test dataset
evaluator_validation = torch.utils.data.DataLoader(test_dataset,batch_size=1) 

## do federated learning between honest clients and malicious clients
# create honest clients 
clients = []
honestidx = []
malidx = []
for index, dataset in enumerate(honest_clients_dataset):
    honestidx.append(index)
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

if num_malicious_clients > 0:
    for index, dataset in enumerate(malicious_datasets):
        malidx.append(index)
        clients.append(
            Client(
                {"id": index+len(honest_clients_dataset),
                "train_data": dataset,
                "val_size": val_size,
                "batch_size": batch_size,
                "is_malicious": True,
                "local_epoch": local_epoch
                }
            )
        )

# shuffle the client
random.shuffle(clients)

print("Clients created...")
print(f"Number of honest clients: {honest_client_numbers}")
print(f"Number of malicious clients: {num_malicious_clients}")

## start Federated Learning cycle:
## set server model 
server_model = CnnModel(input_size=(3,32,32), num_classes=num_classes)

### FedAvg:
if algorithm=="FedAvg":
    ## Set algorithm specific requirements

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
elif algorithm=='Proposed':
    ## Set algorithm specific requirements
    
    ## Open language model
    ## NOTE: If we create the model language model for each client, it will be too much for one computer
    img_text_model, feature_extractor, tokenizer = get_image_to_text_model()

    # evaluator description
    evaluator_description = get_data_description(test_dataset, num_evaluator_sample, img_text_model, feature_extractor, tokenizer)

    for train_round in range(fl_rounds):
        print(f'Round {train_round}...')
        
        ## Build description
        overall_description= ""
        # add evaluator description
        overall_description += f"Evaluator: {evaluator_description}\n"

        ## number of clients to pick
        num_clients_to_pick = int(C * len(clients))
        ## list clients
        round_available_index = np.random.choice(len(clients), num_clients_to_pick, replace=False)
        available_clients = [clients[client_index] for client_index in round_available_index]

        # for each listed available clients sample num_sample of their data and get it's description by language model
        for index,client in enumerate(available_clients):
            client_description = get_data_description(client.raw_train_data , num_sample, img_text_model, feature_extractor, tokenizer)
            overall_description += f"Client {index}: {client_description}\n"

        ## set up the prompt
        prompt = f"""
        Data description: '''{overall_description}'''
        
        [...]
        Comparing the descriptions above, I want you to give me a list of client's ID that have very similar context with the evaluator's context by filling the empty bracket above. No need for explanation. Just fill the bracket with a list of client's ID.
        If there are no clients with similar context, return it as an empty list.
        """
        
        response = get_completion(prompt)
        print(prompt)
        print(f'Response from GPT: \n{response}')
        selected_client = [available_clients[int(x)] for x in response[1:-1].split(',')]

        ## do training

        for client in selected_client:
            ## get updated model from server
            client.model = copy.deepcopy(server_model)
            ## train clients
            client.train(algorithm=algorithm)
        
        ## Update server model
        total_data_points = sum([len(train_clients)for train_clients in selected_client])
        fed_avg_freqs = [len(train_clients)/ total_data_points for train_clients in selected_client]

        global_w = server_model.state_dict()
        for net_id, train_clients in enumerate(selected_client):
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
else:
    pass