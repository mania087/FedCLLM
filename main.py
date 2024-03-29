import numpy as np
import torch
import random
import copy
import itertools
import datetime
import openai 
import logging
import os
import json
import gensim.downloader
import argparse
import torchvision.models as models
import time

from dataloader import create_datasets
from client import Client
from model import CnnModel, ResNet50, FineTunedModel, ResNet18
from utils import test, get_image_to_text_model, get_data_description, get_completion, get_api_key, mkdirs, compare_sentences_score, get_model_combination

## set key
openai.api_key  = get_api_key('backup/gpt_key.txt')

## set seed
np.random.seed(seed=42)
torch.manual_seed(0)

# do the first prompt to make content summary
def make_prompt_summary(description, word_summary_max=15):
    prompt_1 = f"""You are a helpful assistant whose task is to figure what is the theme from a list of descriptions.
    
    Write a summary based on the information provided in the descriptions delimited by triple dashes.

    Use at most {word_summary_max} words and answer in a single sentence. 

    Descriptions: ---{description}---
    """
    return prompt_1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--data_dir', type=str, required=False, default='../../dataset', help='Data directory path')
    parser.add_argument('--dataset', type=str, required=False, default='Food101', help='Dataset name: TinyImagenet, Food101, FMNIST, MNIST, CIFAR10')
    parser.add_argument('--malicious_dataset', type=str, required=False, default='TinyImagenet', help='Dataset name: TinyImagenet, Food101, FMNIST, MNIST, CIFAR10, Cub2011')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--algorithm', type=str, default='Proposed', help='Available algorithm: FedAvg, Proposed, ACS, Shapley')
    parser.add_argument('--num_clients', type=int, default=15, help='Number of clients to simulate')
    parser.add_argument('--num_malicious_clients', type=int, default=5, help='Number of malicious clients to simulate')
    parser.add_argument('--local_epoch', type=int, default=1, help='Number of local epoch for each round')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size for each client')
    parser.add_argument('--rounds', type=int, default=100, help='Number of rounds for FL training')
    parser.add_argument('--C', type=float, default=1.0, help='Percentage of clients available for each round')
    parser.add_argument('--sim_threshold', type=float, default=0.5, help='Threshold for cosine similarity method')
    parser.add_argument('--iid', type=bool, default=True, help='Set data split to IID or non-IID')
    parser.add_argument('--fraction_rank', type=float, default=0.5, help='ACS parameter: Fraction of clients with highest accuracy')
    parser.add_argument('--num_description_sample', type=int, default=10, help='number of data descriptions to sample from each client')
    parser.add_argument('--server_num_description', type=int, default=10, help='number of data descriptions to sample from the server')
    parser.add_argument('--random_select_fract', type=float, default=1.0, help='to select client randomly')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the model')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer for the model: sgd, adam, rmsprop')
    parser.add_argument('--word_summary_max', type=int, default=15, help='maximum number of words for summary, CURRENTLY FOR SERVER')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    
    date_time= datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    
    if args.log_file_name is None:
        argument_path = f'{args.algorithm}_dataset_{args.dataset}_malicious_{args.malicious_dataset}_arguments-{date_time}.json'
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(vars(args), f, indent=4)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    if args.log_file_name is None:
        args.log_file_name = f'{args.algorithm}_dataset_{args.dataset}_malicious_{args.malicious_dataset}_log-{date_time}'
    
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
        
    logger.info("#" * 100)
    
    ## load dataset for clients and malicious clients
    honest_client_numbers = args.num_clients - args.num_malicious_clients

    ## NOTE: somehow handling fraction using num clients and shards are hard, 
    ## There are some consideration to use dirilect
    local_datasets, test_dataset, val_dataset, category = create_datasets(data_path=args.data_dir, 
                                                                dataset_name=args.dataset, 
                                                                num_clients=honest_client_numbers, 
                                                                num_shards=200, 
                                                                iid=args.iid, 
                                                                transform=None, 
                                                                print_count=True)
    
    if args.num_malicious_clients > 0:
        malicious_datasets, _, _= create_datasets(data_path=args.data_dir, 
                                                  dataset_name=args.malicious_dataset, 
                                                  num_clients=args.num_malicious_clients, 
                                                  num_shards=200, 
                                                  separate_validation_data=False,
                                                  iid=args.iid, 
                                                  transform=None, 
                                                  print_count=True)

    # pick honest client datasets
    selected_indices = np.random.choice(len(local_datasets), honest_client_numbers, replace=False)
    honest_clients_dataset = [local_datasets[x] for x in selected_indices]

    # create evaluator & test loader
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size)

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
    
    logger.info(f"Number of honest clients: {honest_client_numbers}")
    logger.info(f"Number of malicious clients: {args.num_malicious_clients}")

    ## start Federated Learning cycle:
    ## set server model 
    #server_model = CnnModel(input_size=(3,32,32), num_classes=num_classes)
    
    #server_model = ResNet50(category.shape[0])
    
    server_model = models.resnet18(pretrained=False)
    server_model.fc = torch.nn.Linear(server_model.fc.in_features, category.shape[0])
    
    #server_model = models.mobilenet_v2(pretrained=False)
    #server_model.classifier[1] = torch.nn.Linear(server_model.last_channel, category.shape[0])
    
    #mobilenet = models.mobilenet_v2(pretrained=True)
    #server_model = FineTunedModel(mobilenet, category.shape[0])
    #for param in server_model.base_model.features.parameters():
    #    param.requires_grad = False
    
    print('number of classes:', category)

    ## set recording metrics
    metrics = {
        "f1": [],
        "acc": [],
        "rec": [],
        "prec": [],
        "loss": []
    }
    
    ## count the time execution
    start_time = time.time()
    
    ### FedAvg:
    logger.info(f"Starting algorithm : {args.algorithm}")
    if args.algorithm=="FedAvg":
        ## Set algorithm specific requirements

        for train_round in range(args.rounds):
            print(f'Round {train_round}...')
            logger.info(f'Round {train_round}...')
            ## number of clients to pick
            num_clients_to_pick = int(args.C * len(clients))
            ## pick clients
            round_selected_clients = np.random.choice(len(clients), num_clients_to_pick, replace=False)
            selected_clients = [clients[client_index] for client_index in round_selected_clients]
            
            # if random_select_fract is not 1.0, select the clients randomly
            if args.random_select_fract < 1.0:
                # select index randomly
                select_index = np.random.choice(len(selected_clients), int(args.random_select_fract*len(selected_clients)), replace=False)
                selected_clients = [clients[client_index] for client_index in select_index]
                
            print(f'Training with {len(selected_clients)} clients...')
            logger.info(f'Training with {len(selected_clients)} clients...')

            for train_clients in selected_clients:
                ## get updated model from server
                train_clients.model = copy.deepcopy(server_model)
                # use multiple GPUs
                train_clients.model = torch.nn.DataParallel(train_clients.model)
                ## train clients
                logger.info(f'Training client {train_clients.id}...')
                train_clients.train(algorithm=args.algorithm, opt=args.opt, lr=args.lr)
            
            ## Update server model
            total_data_points = sum([len(train_clients)for train_clients in selected_clients])
            fed_avg_freqs = [len(train_clients)/ total_data_points for train_clients in selected_clients]

            global_w = server_model.state_dict()
            for net_id, train_clients in enumerate(selected_clients):
                try:
                    net_para = train_clients.model.module.state_dict()
                except AttributeError:
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
            test_loss, results = test(server_model, test_loader)
            print(f"loss:{test_loss.item()}, metrics:{results}")
            logger.info(f"loss:{test_loss.item()}, metrics:{results}")
            ## metrics saved
            metrics["loss"].append(test_loss.item())
            metrics["acc"].append(results["acc"])
            metrics["rec"].append(results["rec"])
            metrics["f1"].append(results["f1"])
            metrics["prec"].append(results["prec"])
            
            ## save model
            torch.save(server_model.state_dict(), f'models/{args.algorithm}_{date_time}.pt')
    
    elif args.algorithm == 'ACS':
        ## ACS algorithm (ASSUME server validation is labeled)
        # for the first round, use all the clients
        # for the rest of the round , use the ranked client 
        
        # initialize all client accuracy as 0.0
        Client_accuracy_record = {}
        for client in clients:
            Client_accuracy_record[client.id] = 0.0
        
        for train_round in range(args.rounds):
            print(f'Round {train_round}...')
            logger.info(f'Round {train_round}...')
            
            ## number of clients to pick
            num_clients_to_pick = int(args.C * len(clients))
            ## pick clients
            round_selected_clients = np.random.choice(len(clients), num_clients_to_pick, replace=False)
            selected_clients = [clients[client_index] for client_index in round_selected_clients]
            print(f'Training with {len(selected_clients)} clients...')
            logger.info(f'Training with {len(selected_clients)} clients...')
            
            # if round is not zero, use the ranked client
            if train_round > 0:
                # change the selected clients
                # get the order from highest to lowest from client accuracy record
                client_id_highest = sorted(Client_accuracy_record, key=Client_accuracy_record.get, reverse=True)
                
                # check if the client with id in the list are on the selected client list
                temp = []
                list_of_current_round_id = [client.id for client in selected_clients]
                maximum_length_add = int(args.fraction_rank*len(selected_clients))
                for id in client_id_highest:
                    if id in list_of_current_round_id:
                        loc_id = list_of_current_round_id.index(id)
                        temp.append(selected_clients[loc_id])
                    # if it is already full, break
                    if len(temp) == maximum_length_add:
                        break
                        
                # change temp to selected list
                selected_clients = temp
            
            for train_clients in selected_clients:
                ## get updated model from server
                train_clients.model = copy.deepcopy(server_model)
                # use multiple GPUs
                train_clients.model = torch.nn.DataParallel(train_clients.model)
                ## train clients
                logger.info(f'Training client {train_clients.id}...')
                # ACS used the same training method as FedAvg
                train_clients.train(algorithm="FedAvg", opt=args.opt, lr=args.lr)
                
                # update the client accuracy on server evaluation data
                Client_accuracy_record[train_clients.id] = test(train_clients.model, val_loader)
            
            ## Update server model
            total_data_points = sum([len(train_clients)for train_clients in selected_clients])
            fed_avg_freqs = [len(train_clients)/ total_data_points for train_clients in selected_clients]

            global_w = server_model.state_dict()
            for net_id, train_clients in enumerate(selected_clients):
                try:
                    net_para = train_clients.model.module.state_dict()
                except AttributeError:
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
            test_loss, results = test(server_model, test_loader)
            print(f"loss:{test_loss.item()}, metrics:{results}")
            logger.info(f"loss:{test_loss.item()}, metrics:{results}")
            ## metrics saved
            metrics["loss"].append(test_loss.item())
            metrics["acc"].append(results["acc"])
            metrics["rec"].append(results["rec"])
            metrics["f1"].append(results["f1"])
            metrics["prec"].append(results["prec"])
            
            ## save model
            torch.save(server_model.state_dict(), f'models/{args.algorithm}_{date_time}.pt')
            
        
    elif args.algorithm == 'Proposed':
        ## Do the double prompt
        ## Prompt 1 : Get the summary of overall data descriptions for evaluators and clients
        ## Prompt 2 : Get the list of clients that have similar context with the evaluator
        
        ## Open image to text model
        ## NOTE: If we create the model language model for each client, it will be too much for one computer
        img_text_model, feature_extractor, tokenizer = get_image_to_text_model()  
        
        for train_round in range(args.rounds):
            # evaluator description
            evaluator_description = get_data_description(val_dataset, args.server_num_description, img_text_model, feature_extractor, tokenizer)
            prompt_survey = make_prompt_summary(evaluator_description, args.word_summary_max)
            evaluator_survey = get_completion(prompt_survey)
            
            print(f'Round {train_round}...')
            logger.info(f'Round {train_round}...')
            
            ## number of clients to pick
            num_clients_to_pick = int(args.C * len(clients))
            ## list clients
            round_available_index = np.random.choice(len(clients), num_clients_to_pick, replace=False)
            available_clients = [clients[client_index] for client_index in round_available_index]
            
            # for each listed available clients sample num_sample of their data and get it's description by language model
            # and then perform prompt 1
            client_summaries = {}
            overall_description= ""
            for index, client in enumerate(available_clients):
                description = get_data_description(client.train_dataset , args.num_description_sample, img_text_model, feature_extractor, tokenizer)
                
                prompt_client = make_prompt_summary(description)
                client_summaries[client.id] = get_completion(prompt_client)
                overall_description += f"Client {client.id}: {client_summaries[client.id]}\n"
                
                print(f"Client {client.id}: {client_summaries[client.id]} ")
                logger.info(f"Client {client.id}: {client_summaries[client.id]} ")
            
            prompt_selection = f"""You are a helpful assistant whose task is to select the clients that have similar context with the evaluator's context.
            
            Please compare each client description in Data description with the Evaluator Description and provide the reasons.
            
            Answer in a python dictionary format. For example, if you think Client 1 and Client 2 have similar context with the evaluator's context, you can write it as {{"reasons": "...", "selected": [1,2]}}.
            
            The Evaluator description is marked with triple dashes and the Data description is marked with triple ticks.
            
            Evaluator description: ---{evaluator_survey}---
            
            Data description: '''{overall_description}'''
            """
            print(prompt_selection)
            logger.info(prompt_selection)
            
            prompt_answer = get_completion(prompt_selection)
            
            # the prompt answer
            print(prompt_answer)
            logger.info(prompt_answer)
            
            selected_honest_client = json.loads(prompt_answer)
            
            selected_clients = [client for client in available_clients if int(client.id) in selected_honest_client["selected"]]
            # the parsed answers
            print(f'the selected clients are: {selected_honest_client["selected"]}')
            logger.info(f'the selected clients are: {selected_honest_client["selected"]}')
            ## do training
            for client in selected_clients:
                ## get updated model from server
                client.model = copy.deepcopy(server_model)
                # use multiple GPUs
                client.model = torch.nn.DataParallel(client.model)
                ## train clients
                logger.info(f'Training client {client.id}...')
                # the training mechanism is the same as FedAvg
                client.train(algorithm="FedAvg", opt=args.opt, lr=args.lr)
            
            ## Update server model
            total_data_points = sum([len(train_clients)for train_clients in selected_clients])
            fed_avg_freqs = [len(train_clients)/ total_data_points for train_clients in selected_clients]

            global_w = server_model.state_dict()
            for net_id, train_clients in enumerate(selected_clients):
                try:
                    net_para = train_clients.model.module.state_dict()
                except AttributeError:
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
            test_loss, results = test(server_model, test_loader)
            print(f"loss:{test_loss.item()}, metrics:{results}")
            logger.info(f"loss:{test_loss.item()}, metrics:{results}")
            ## metrics saved
            metrics["loss"].append(test_loss.item())
            metrics["acc"].append(results["acc"])
            metrics["rec"].append(results["rec"])
            metrics["f1"].append(results["f1"])
            metrics["prec"].append(results["prec"])
            
            torch.save(server_model.state_dict(), f'models/{args.algorithm}_{date_time}.pt')
            
    elif args.algorithm == 'Proposed_old':
        ## Set algorithm specific requirements
        
        ## Open image to text model
        ## NOTE: If we create the model language model for each client, it will be too much for one computer
        img_text_model, feature_extractor, tokenizer = get_image_to_text_model()

        for train_round in range(args.rounds):
            # evaluator description
            evaluator_description = get_data_description(val_dataset, args.server_num_description, img_text_model, feature_extractor, tokenizer)
            
            print(f'Round {train_round}...')
            logger.info(f'Round {train_round}...')
            
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
                client_description = get_data_description(client.train_dataset , args.num_description_sample, img_text_model, feature_extractor, tokenizer)
                #overall_description += f"Client {index}: {client_description}\n"
                ## for debugging first
                print(f'Client {client.id} description:{client_description}')
                logger.info(f'Client {client.id} description:{client_description}')
                overall_description += f"Client {client.id}: {client_description}\n"

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
            logger.info(f'Response from GPT, selected clients: {response}...')
            # selected_client = [available_clients[int(x)] for x in response[1:-1].split(',')]
            ## for debugging first
            parsed_reponse =  [int(x) for x in response[1:-1].split(',')]
            selected_client = [client for client in available_clients if int(client.id) in parsed_reponse]
            
            ## do training

            for client in selected_client:
                ## get updated model from server
                client.model = copy.deepcopy(server_model)
                # use multiple GPUs
                client.model = torch.nn.DataParallel(client.model)
                ## train clients
                logger.info(f'Training client {client.id}...')
                client.train(algorithm="FedAvg", opt=args.opt, lr=args.lr)
            
            ## Update server model
            total_data_points = sum([len(train_clients)for train_clients in selected_client])
            fed_avg_freqs = [len(train_clients)/ total_data_points for train_clients in selected_client]

            global_w = server_model.state_dict()
            for net_id, train_clients in enumerate(selected_client):
                try:
                    net_para = train_clients.model.module.state_dict()
                except AttributeError:
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
            test_loss, results = test(server_model, test_loader)
            print(f"loss:{test_loss.item()}, metrics:{results}")
            logger.info(f"loss:{test_loss.item()}, metrics:{results}")
            ## metrics saved
            metrics["loss"].append(test_loss.item())
            metrics["acc"].append(results["acc"])
            metrics["rec"].append(results["rec"])
            metrics["f1"].append(results["f1"])
            metrics["prec"].append(results["prec"])
            
            torch.save(server_model.state_dict(), f'models/{args.algorithm}_{date_time}.pt')
    
    ### Other algorithm
    elif args.algorithm=='Cosine':
        ## Open image to text model
        ## NOTE: If we create the model language model for each client, it will be too much for one computer
        img_text_model, feature_extractor, tokenizer = get_image_to_text_model()
        
        ## load pretrained word2vec
        word_model = gensim.downloader.load('word2vec-google-news-300')
        # word_model = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        
        for train_round in range(args.rounds):
            # evaluator description
            evaluator_description = get_data_description(val_dataset, args.num_description_sample, img_text_model, feature_extractor, tokenizer)
            
            print(f'Round {train_round}...')
            logger.info(f'Round {train_round}...')
            
            ## Print evaluator description
            print(f'Evaluator description:{evaluator_description}')
            logger.info(f'Evaluator description:{evaluator_description}')

            ## number of clients to pick
            num_clients_to_pick = int(args.C * len(clients))
            ## list clients
            round_available_index = np.random.choice(len(clients), num_clients_to_pick, replace=False)
            available_clients = [clients[client_index] for client_index in round_available_index]
            
            # for each listed available clients sample num_sample of their data and get it's description by language model
            selected_client = []
            selected_client_index = []
            for index,client in enumerate(available_clients):
                client_description = get_data_description(client.train_dataset , args.num_description_sample, img_text_model, feature_extractor, tokenizer)
                #print(f'Client {index} description:{client_description}')
                #logger.info(f'Client {index} description:{client_description}')
                ## for debugging
                print(f'Client {client.id} description:{client_description}')
                logger.info(f'Client {client.id} description:{client_description}')
                
                # get distance
                client_similarity = compare_sentences_score(evaluator_description,client_description,word_model)
                avg_similarity = np.average(np.array(client_similarity))
                
                #print(f'Similarity of client {index}:{avg_similarity}')
                #logger.info(f'Similarity of client {index}:{avg_similarity}')
                ## for debugging
                print(f'Similarity of client {client.id}:{avg_similarity}')
                logger.info(f'Similarity of client {client.id}:{avg_similarity}')
                # threshold for including the client
                if avg_similarity > args.sim_threshold:
                    selected_client.append(client)
                    selected_client_index.append(index)
                    
            print(f'Included Clients:{selected_client_index}')
            logger.info(f'Included Clients:{selected_client_index}')
                    
            ## do training
            for client in selected_client:
                ## get updated model from server
                client.model = copy.deepcopy(server_model)
                ## train clients
                logger.info(f'Training client {client.id}...')
                client.train(algorithm="FedAvg", opt=args.opt, lr=args.lr)
            
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
            test_loss, results = test(server_model, test_loader)
            print(f"loss:{test_loss.item()}, metrics:{results}")
            logger.info(f"loss:{test_loss.item()}, metrics:{results}")
            ## metrics saved
            metrics["loss"].append(test_loss.item())
            metrics["acc"].append(results["acc"])
            metrics["rec"].append(results["rec"])
            metrics["f1"].append(results["f1"])
            metrics["prec"].append(results["prec"])
            
            torch.save(server_model.state_dict(), f'models/{args.algorithm}_{date_time}.pt')
    ## Shapley Value ##
    elif args.algorithm=='Shapley':
        ## Set algorithm specific requirements

        for train_round in range(args.rounds):
            print(f'Round {train_round}...')
            logger.info(f'Round {train_round}...')
            ## number of clients to pick
            num_clients_to_pick = int(args.C * len(clients))
            ## pick clients
            round_selected_clients = np.random.choice(len(clients), num_clients_to_pick, replace=False)
            selected_clients = [clients[client_index] for client_index in round_selected_clients]
            print(f'Training with {len(selected_clients)} clients...')
            logger.info(f'Training with {len(selected_clients)} clients...')
            
            # get availible local updates first
            local_models= []
            for train_clients in selected_clients:
                ## get updated model from server
                train_clients.model = copy.deepcopy(server_model)
                ## train clients
                logger.info(f'Training client {train_clients.id}...')
                train_clients.train(algorithm="FedAvg", opt=args.opt, lr=args.lr)
                ## get local model
                local_models.append(train_clients.model.state_dict())
            
            ## Count the frequencies
            total_data_points = sum([len(train_clients)for train_clients in selected_clients])
            fed_avg_freqs = [len(train_clients)/ total_data_points for train_clients in selected_clients]
            
            ## get the permutation and try combination of local models and see which one is the best accordnnig to the evaluator test dataset
            best_model = None
            best_accuracy = 0.0
            template_model = copy.deepcopy(server_model)
            for i in range(0,len(local_models)):
                model_indexes = list(range(0,len(local_models)))
                list_of_permutations = list(itertools.combinations(model_indexes, i+1))
                
                for permutation in list_of_permutations:
                    # get the model parameters
                    model_parameters = []
                    for index_model in permutation:
                        model_parameters.append(local_models[index_model])
                    # get the combination parameters
                    combination_weight = get_model_combination(model_parameters, template_model)
                    # load the combination parameters
                    template_model.load_state_dict(combination_weight)
                    ## test model
                    _, results = test(template_model, test_loader)
                    
                    print(f"model of {permutation}, metrics:{results}")
                    logger.info(f"model of {permutation}, metrics:{results}")
                    
                    if results["acc"]>best_accuracy:
                        # replace the best model
                        best_accuracy = results["acc"]
                        best_model = copy.deepcopy(template_model)
            
            ## load global model with the best model
            server_model.load_state_dict(best_model.state_dict())

            ## test model
            test_loss, results = test(server_model, test_loader)
            print(f"loss:{test_loss.item()}, metrics:{results}")
            logger.info(f"loss:{test_loss.item()}, metrics:{results}")
            ## metrics saved
            metrics["loss"].append(test_loss.item())
            metrics["acc"].append(results["acc"])
            metrics["rec"].append(results["rec"])
            metrics["f1"].append(results["f1"])
            metrics["prec"].append(results["prec"])
            
            ## save model
            torch.save(server_model.state_dict(), f'models/{args.algorithm}_{date_time}.pt')
    else:
        pass
    
    # print execution time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time}")
    logger.info(f"Total execution time: {total_time}")
    # print results in the end
    print(f"Global testing results : {metrics}")
    logger.info(f"Global testing results : {metrics}")