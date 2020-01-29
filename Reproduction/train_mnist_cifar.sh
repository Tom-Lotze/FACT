#!/bin/bash
#SBATCH --job-name=train_mnist_cifar
#SBATCH --time=20:00:00
#SBATCH --mem=20G
#SBATCH --partition=gpu_shared_course

echo "Running" | mail $USER

python train_mnist_cifar.py

echo "Done" | mail $USER