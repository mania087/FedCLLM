
# FedCLLM: Federated client selection assisted large language model utilizing domain description

## Abstract
Federated Learning has become an emerging topic since the rise of privacy regulation regarding personal data protection and sensitivity. It provides a decentralized training approach to train a global model between a server and multiple clients while ensuring client data confidentiality. However, in practical scenarios, there are malicious clients within a large pool of client candidates, and selecting trustworthy honest clients becomes a crucial problem. Some previous works tried to solve the problem by exchanging client data for comparison and using a labelled dataset for evaluating client models to select honest clients. However, they pose a limitation when the server possesses unlabelled data from other sources and has limited or unavailable resources to label the data. To address the issue, this work proposes FedCLLM, a novel approach using Large Language Models (LLM) proficiency in semantic tasks on text-based data to compare client and server domain descriptions summary in a text format and assess their similarity. Experiments on popular benchmark datasets show that FedCLLM effectively distinguishes honest clients from potentially malicious ones and outperforms other previous works in terms of performance.


## Installation
Install the requirements with:
```
pip install -r requirements.txt
```

## Run the code
An example to run the file can be found on  `run.sh`.
```
bash run.sh
```

To rerun the experiments without changing anything about the config files, please place the complete dataset download into a folder called `data` in the main directory of the repository.

## Configuration

| Parameter | Description |
| ------ | ------ |
| `log_file_name` | The log file name |
| `data_dir` | Data directory path |
| `dataset` | Dataset for honest clients |
| `malicious_dataset` | Dataset for malicious clients |
| `logdir` | Experiment log directory path |
| `modeldir` | Model directory path |
| `algorithm` | Available algorithms: FedAvg, FedAvgM, Proposed, ACS |
| `num_clients` | Number of clients to simulate |
| `num_malicious_clients` | Number of malicious clients to simulate |
| `local_epoch` | Number of local epochs for each round |
| `batch_size` | Batch size for each client |
| `rounds` | Number of rounds for FL training |
| `C` | Percentage of clients available for each round |
| `iid` | Set data split to IID or non-IID |
| `fraction_rank` | ACS parameter: Fraction of clients with the highest accuracy |
| `num_description_sample` | Number of data descriptions to sample from each client |
| `server_num_description` | Number of data descriptions to sample from the server |
| `random_select_fract` | Fraction of clients selected randomly |
| `server_momentum` | FedAvgM parameter: Momentum for the server |
| `lr` | Learning rate for the model |
| `opt` | Optimizer for the model: SGD, Adam, RMSprop |
| `word_summary_max` | Maximum number of words for summary (Currently for the server) |
| `non_iid_mode` | Mode for non-IID: Dirichlet, Shard |
| `dirichlet_alpha` | Alpha parameter for the Dirichlet distribution |
| `top_p` | Subset of tokens to consider for LLM: Client selection step |
| `temperature` | Controls the randomness of LLM: Client selection step |
| `summarization_top_p` | Subset of tokens to consider for LLM: Summarization |
| `summarization_temperature` | Controls the randomness of LLM: Summarization |


## Cite as

```
@article{iwan2025fedcllm,
  title={FedCLLM: Federated client selection assisted large language model utilizing domain description},
  author={Iwan, Ignatius and Tanjung, Sean Yonathan and Yahya, Bernardo Nugroho and Lee, Seok-Lyong},
  journal={Internet of Things},
  volume={30},
  pages={101506},
  year={2025},
  publisher={Elsevier}
}
```