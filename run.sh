#!/bin/bash
dataset='CIFAR10'
malicious_dataset='MNIST'
num_clients=15
num_malicious_clients=5
local_epoch=1
batch_size=64
num_description_sample=10
server_num_description=20
rounds=100
C=1.0 
word_summary_max=15
lr=0.001
opt="adam"
t_round=60
non_iid_mode="dirichlet"
dirichlet_alpha=1.0
top_p=0.1
temperature=0.2
summarization_top_p=0.2
summarization_temperature=0.1

python main.py --algorithm "Proposed" --top_p $top_p --temperature $temperature --summarization_top_p $summarization_top_p --summarization_temperature $summarization_temperature --non_iid_mode $non_iid_mode --dirichlet_alpha $dirichlet_alpha --dataset $dataset --malicious_dataset $malicious_dataset --num_clients $num_clients --num_malicious_clients $num_malicious_clients --local_epoch $local_epoch --batch_size $batch_size --num_description_sample $num_description_sample --server_num_description $server_num_description --rounds $rounds --C $C --word_summary_max $word_summary_max --lr $lr --opt $opt