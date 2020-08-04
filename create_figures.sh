#!/usr/bin/env bash


## Split MNIST (task- and class-incremental learning)

# - hyperparameter gridsearch
./compare_MNIST_hyperParams.py --scenario=task --seed=11                 #--> Fig. S1 (top)
./compare_MNIST_hyperParams.py --scenario=class --seed=11                #--> Fig. S1 (bottom)
# - compare methods
./compare_MNIST.py --scenario=task --seed=12 --n-seeds=20                #--> Fig. 3B
./compare_MNIST.py --scenario=class --seed=12 --n-seeds=20               #--> Fig. 3C
# - analysis of efficiency and robustness of replay
./compare_MNIST_replay.py --scenario=task --seed=12 --n-seeds=20         #--> Fig. 4A,B (left)
./compare_MNIST_replay.py --scenario=class --seed=12 --n-seeds=20        #--> Fig. 4A,B (right)


## Permuted MNIST with 100 permutations

# - hyperparameter gridsearch
./compare_permMNIST100_hyperParams.py --seed=11                          #--> Fig. S2
# - compare methods
./compare_permMNIST100.py --seed=12 --n-seeds=5                          #--> Fig. 6
# - addition- and ablation-experiments
./compare_permMNIST100_bir.py --seed=12 --n-seeds=5                      #--> Fig. 8A


## Split CIFAR-100 (task- and class-incremental learning)

# - pre-train convolutional layers ("e100") on CIFAR-10
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100
# - hyperparameter gridsearch
./compare_CIFAR100_hyperParams.py --scenario=task --convE-ltag=e100 --seed=11       #--> Fig. S3 (top)
./compare_CIFAR100_hyperParams.py --scenario=class --convE-ltag=e100 --seed=11      #--> Fig. S3 (bottom)
# - compare methods
./compare_CIFAR100.py --scenario=task --convE-ltag=e100 --seed=12 --n-seeds=10      #--> Fig. 7B
./compare_CIFAR100.py --scenario=class --convE-ltag=e100 --seed=12 --n-seeds=10     #--> Fig. 7C
# - train an embedding network ("f20") on CIFAR-100 for evaluating generator performance
./main_pretrain.py --experiment=CIFAR100 --epochs=20 --augment --pre-convE --convE-ltag=e100 --freeze-convE --full-stag=f20
# - addition- and ablation-experiments
./compare_CIFAR100_bir.py --scenario=task --convE-ltag=e100 --seed=12 --n-seeds=10   #--> Fig. 8B
./compare_CIFAR100_bir.py --scenario=class --convE-ltag=e100 --seed=12 --n-seeds=10  #--> Fig. 8C
# - analysis of quality replay
./compare_CIFAR100_bir.py --scenario=class --convE-ltag=e100 --seed=12 --n-seeds=10 --eval-gen --eval-tag=f20 #--> Fig. 9


# NOTE: for "compare_permMNIST100_bir.py" and "compare_CIFAR100_bir.py", the selected values for the hyper-parameter
#       [dg_prop] for the various replay-variants are hard-coded within these scrips. For all other scripts, all
#       hyper-parameters can be changed from their selected default value by specifying options when calling the script.
