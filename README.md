FedAH: Federated Learning with Attentive Hierarchy on Non-IID Data

## Dependencies

The code requires Python >= 3.6 and PyTorch >= 1.2.0. To install the other dependencies: `pip3 install -r requirements.txt`.

## Data

This code uses the CIFAR10, CIFAR100, Federated Extended MNIST (FEMNIST), MNIST, and Sentiment140 (Sent140) datasets.

The CIFAR10, CIFAR100 AND MNIST datasets are downloaded automatically by the torchvision package. 
FEMNIST and Sent140 are provided by the LEAF repository, which should be downloaded from https://github.com/TalwalkarLab/leaf/ and placed in `FedAH/`. 
Then the raw FEMNIST and Sent140 datasets can be downloaded by following the instructions in LEAF. 
In order to generate the versions of these datasets that we use the paper, we use the following commands from within `FedAH/data/sent140/` and `FedAH/data/femnist/`, respectively:

Sent140: `./preprocess.sh -s niid --sf 0.3 -k 50 -tf 0.8 -t sample`

FEMNIST: `./preprocess.sh -s niid --sf 0.5 -k 50 -tf 0.8 -t sample`

For FEMNIST, we re-sample and re-partition the data to increase its heterogeneity. In order to do so, first navigate to `FedAH/`, then execute 

`mv my_sample.py data/femnist/data/`

`cd data/femnist/data/`

`python my_sample.py`

## Usage

FedAH is run using a command of the following form:

`python main_fedah.py --alg fedah --dataset [dataset] --num_users [num_users] --model [model] --model [model] --shard_per_user [shard_per_user] --frac [frac] --local_bs [local_bs] --lr [lr] --epochs [epochs] --local_ep [local_ep] --local_rep_ep [local_rep_ep] --gpu [gpu] --seed [seed]`

Explanation of parameters:

- `alg` : algorithm to run
- `dataset` : dataset, may be `cifar10`, `cifar100`, `femnist`, `mnist`, `sent140`
- `num_users` : number of users
- `model` : for the CIFAR datasets, we use `cnn`, for the MNIST datasets, we use `mlp`, and for `sent140`, we use `res`
- `num_classes` : total number of classes
- `shard_per_user` : number of classes per user (specific to CIFAR datasets and MNIST)
- `frac` : fraction of participating users in each round (for all experiments we use 0.1)
- `local_bs` : batch size used locally by each user
- `lr` : learning rate
- `epochs` : total number of communication rounds
- `local_ep` : total number of local epochs
- `local_rep_ep` : number of local epochs to execute for the representation (specific to Fedah)
- `gpu` : GPU ID
- `seed`: random seed

A full list of configuration parameters and their descriptions are given in `utils/options.py`.
For examples of running FedAH as well as the FL baselines we compare against, please see the executable files in `scripts/`, which recover the results from the paper.


# Acknowledgements

Much of the code in this repository was adapted from code in the repository https://github.com/lgcollins/FedRep.git
