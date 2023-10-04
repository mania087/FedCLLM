import numpy as np
import torch
import random
import copy
import datetime
import openai 
import logging
import os
import json
import gensim.downloader
import argparse

from dataloader import create_datasets
from client import Client
from model import CnnModel, ResNet50
from utils import test, get_image_to_text_model, get_data_description, get_completion, get_api_key, mkdirs, compare_sentences_score

## set key
openai.api_key  = get_api_key('backup/gpt_key.txt')

## set seed
np.random.seed(seed=42)
torch.manual_seed(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--algorithm', type=str, default='Proposed', help='Available algorithm: FedAvg, Proposed')
    parser.add_argument('--num_clients', type=int, default=15, help='Number of clients to simulate')
    parser.add_argument('--num_malicious_clients', type=int, default=5, help='Number of malicious clients to simulate')
    parser.add_argument('--n_class', type=int, default=10, help='Number of available classes')
    parser.add_argument('--local_epoch', type=int, default=1, help='Number of local epoch for each round')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size for each client')
    parser.add_argument('--rounds', type=int, default=100, help='Number of rounds for FL training')
    parser.add_argument('--C', type=float, default=1.0, help='Percentage of clients available for each round')
    parser.add_argument('--iid', type=bool, default=True, help='Set data split to IID or non-IID')
    parser.add_argument('--num_description_sample', type=int, default=10, help='number of data descriptions to sample from each client and the server')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    
    date_time= datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    
    ## load dataset for clients and malicious clients
    honest_client_numbers = args.num_clients - args.num_malicious_clients

    ## NOTE: somehow handling fraction using num clients and shards are hard, 
    ## There are some consideration to use dirilect
    local_datasets, test_dataset= create_datasets(data_path='data', 
                                                dataset_name='CIFAR10', 
                                                num_clients=honest_client_numbers, 
                                                num_shards=200, 
                                                iid=args.iid, 
                                                transform=None, 
                                                print_count=True)
    if args.num_malicious_clients > 0:
        malicious_datasets, _= create_datasets(data_path='data', 
                                               dataset_name='MNIST', 
                                               num_clients=args.num_malicious_clients, 
                                               num_shards=200, 
                                               iid=args.iid, 
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
    for index, dataset in enumerate(honest_clients_dataset):
        clients.append(
            Client(
                {"id": index,
                "train_data": dataset,
                "val_size": 0.0,
                "batch_size": args.batch_size,
                "is_malicious": False,
                "local_epoch": args.local_epoch
                }
            )
        )

    if args.num_malicious_clients > 0:
        for index, dataset in enumerate(malicious_datasets):
            clients.append(
                Client(
                    {"id": index+len(honest_clients_dataset),
                    "train_data": dataset,
                    "val_size": 0.0,
                    "batch_size": args.batch_size,
                    "is_malicious": True,
                    "local_epoch": args.local_epoch
                    }
                )
            )

    # shuffle the client
    random.shuffle(clients)

    print("Clients created...")
    print(f"Number of honest clients: {honest_client_numbers}")
    print(f"Number of malicious clients: {args.num_malicious_clients}")
    
    ## start Federated Learning cycle:
    ## set server model 
    #server_model = CnnModel(input_size=(3,32,32), num_classes=num_classes)
    server_model = ResNet50(args.n_class)

    ### FedAvg:
    if args.algorithm=="FedAvg":
        ## Set algorithm specific requirements

        for train_round in range(args.rounds):
            print(f'Round {train_round}...')
            ## number of clients to pick
            num_clients_to_pick = int(args.C * len(clients))
            ## pick clients
            round_selected_clients = np.random.choice(len(clients), num_clients_to_pick, replace=False)
            selected_clients = [clients[client_index] for client_index in round_selected_clients]
            print(f'Training with {len(selected_clients)} clients...')

            for train_clients in selected_clients:
                ## get updated model from server
                train_clients.model = copy.deepcopy(server_model)
                ## train clients
                train_clients.train(algorithm=args.algorithm)
            
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
            
            ## save model
            torch.save(server_model.state_dict(), f'models/{args.algorithm}_{date_time}.pt')

    
    elif args.algorithm=='Proposed':
        ## Set algorithm specific requirements
        
        ## Open image to text model
        ## NOTE: If we create the model language model for each client, it will be too much for one computer
        img_text_model, feature_extractor, tokenizer = get_image_to_text_model()


        for train_round in range(args.rounds):
            # evaluator description
            evaluator_description = get_data_description(test_dataset, args.num_description_sample, img_text_model, feature_extractor, tokenizer)
            
            print(f'Round {train_round}...')
            
            ## Build description
            overall_description= ""
            # add evaluator description
            overall_description += f"Evaluator: {evaluator_description}\n"

            ## number of clients to pick
            num_clients_to_pick = int(args.C * len(clients))
            ## list clients
            round_available_index = np.random.choice(len(clients), num_clients_to_pick, replace=False)
            available_clients = [clients[client_index] for client_index in round_available_index]

            # for each listed available clients sample num_sample of their data and get it's description by language model
            for index,client in enumerate(available_clients):
                client_description = get_data_description(client.raw_train_data , args.num_description_sample, img_text_model, feature_extractor, tokenizer)
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
                client.train(algorithm=args.algorithm)
            
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
            
            torch.save(server_model.state_dict(), f'models/{args.algorithm}_{date_time}.pt')
    
    ### Other algorithm
    elif args.algorithm=='Cosine':
        ## Open image to text model
        ## NOTE: If we create the model language model for each client, it will be too much for one computer
        img_text_model, feature_extractor, tokenizer = get_image_to_text_model()
        
        similariy_threshold = 0.5
        
        ## load pretrained word2vec
        word_model = gensim.downloader.load('word2vec-google-news-300')
        # word_model = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        
        for train_round in range(args.rounds):
            # evaluator description
            evaluator_description = get_data_description(test_dataset, args.num_description_sample, img_text_model, feature_extractor, tokenizer)
            
            print(f'Round {train_round}...')
            
            ## Print evaluator description
            print(f'Evaluator description:{evaluator_description}')

            ## number of clients to pick
            num_clients_to_pick = int(args.C * len(clients))
            ## list clients
            round_available_index = np.random.choice(len(clients), num_clients_to_pick, replace=False)
            available_clients = [clients[client_index] for client_index in round_available_index]
            
            # for each listed available clients sample num_sample of their data and get it's description by language model
            selected_client = []
            selected_client_index = []
            for index,client in enumerate(available_clients):
                client_description = get_data_description(client.raw_train_data , args.num_description_sample, img_text_model, feature_extractor, tokenizer)
                print(f'Client {index} description:{client_description}')
                # get distance
                client_similarity = compare_sentences_score(evaluator_description,client_description,word_model)
                avg_similarity = np.average(np.array(client_similarity))
                
                print(f'Similarity of client {index}:{avg_similarity}')
                logger.info(f'Similarity of client {index}:{avg_similarity}')
                # threshold for including the client
                if avg_similarity > similariy_threshold:
                    selected_client.append(client)
                    selected_client_index.append(index)
                    
            print(f'Included Clients:{selected_client_index}')
                    
            ## do training
            for client in selected_client:
                ## get updated model from server
                client.model = copy.deepcopy(server_model)
                ## train clients
                client.train(algorithm=args.algorithm)
            
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
            
            torch.save(server_model.state_dict(), f'models/{args.algorithm}_{date_time}.pt')
    else:
        pass