#!/bin/bash

#SBATCH --job-name=vit_with_pruning_importance_test
#SBATCH --mem-per-cpu=3GB
##SBATCH --output=../out_train/output_%j.txt
#SBATCH --output=output_%j.txt
#SBATCH --chdir=/rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=3
##SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
##SBATCH --mail-user=c.dutoit@student.maastrichtuniversity.nl
#SBATCH --mail-user=chrisspamtopherdt@gmail.com

## Load the python interpreter
module load gcc
module load python
module load cuda/11.0
module load cudnn/8.0.5

## Install libraries
python3 -m pip install --user torch
python3 -m pip install --user numpy
python3 -m pip install --user argparse
python3 -m pip install --user torchvision
python3 -m pip install --user tqdm
#python3 -m pip install --user transformers
python3 -m pip install --user -e git+git://github.com/chrisdt1998/transformers.git@main
python3 -m pip install --user datasets
python3 -m pip install --user sklearn
python3 -m pip install --user tensorboardX
python3 -m pip install --user matplotlib
python3 -m pip install --user opendatasets


## Run code
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/find_head_mask_smart_deit.py --experiment_id=threshold_exp_cifar100_global_pruning --iteration_id=threshold_88 --pruning_threshold=0.88 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/find_head_mask_smart_deit.py --experiment_id=threshold_exp_cifar100_global_pruning --iteration_id=threshold_90 --pruning_threshold=0.90 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/find_head_mask_smart_deit.py --experiment_id=threshold_exp_cifar100_global_pruning --iteration_id=threshold_92 --pruning_threshold=0.92 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/find_head_mask_smart_deit.py --experiment_id=threshold_exp_cifar100_global_pruning --iteration_id=threshold_94 --pruning_threshold=0.94 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/find_head_mask_smart_deit.py --experiment_id=threshold_exp_cifar100_global_pruning --iteration_id=threshold_96 --pruning_threshold=0.96 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/find_head_mask_smart_deit.py --experiment_id=threshold_exp_cifar100_global_pruning --iteration_id=threshold_98 --pruning_threshold=0.98 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/find_head_mask_smart_deit.py --experiment_id=threshold_exp_cifar100_global_pruning --iteration_id=threshold_99 --pruning_threshold=0.99 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/find_head_mask_smart_deit.py --experiment_id=threshold_exp_cifar100_global_pruning --iteration_id=threshold_999 --pruning_threshold=0.999 --dataset_name=cifar100 --prune_whole_layers

#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/full_train_with_masks.py --dataset_name cifar10 --experiment_id threshold_experiments_cifar10_global_pruning --iteration_id None threshold_88 threshold_90 threshold_92 threshold_94 threshold_96 threshold_98 threshold_99 threshold_999
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/full_train_with_masks.py --experiment_id threshold_experiments_cifar100 --iteration_id None

#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/one_train_masks.py --experiment_id=threshold_exp_one_train_cifar100 --iteration_id=threshold_78 --pruning_threshold=0.78 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/one_train_masks.py --experiment_id=threshold_exp_one_train_cifar100 --iteration_id=threshold_81 --pruning_threshold=0.81 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/one_train_masks.py --experiment_id=threshold_exp_one_train_cifar100 --iteration_id=threshold_84 --pruning_threshold=0.84 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/one_train_masks.py --experiment_id=threshold_exp_one_train_cifar100 --iteration_id=threshold_87 --pruning_threshold=0.87 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/one_train_masks.py --experiment_id=threshold_exp_one_train_cifar100 --iteration_id=threshold_90 --pruning_threshold=0.90 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/one_train_masks.py --experiment_id=threshold_exp_one_train_cifar100 --iteration_id=threshold_93 --pruning_threshold=0.93 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/one_train_masks.py --experiment_id=threshold_exp_one_train_cifar100 --iteration_id=threshold_96 --pruning_threshold=0.96 --dataset_name=cifar100 --prune_whole_layers
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/one_train_masks.py --experiment_id=threshold_exp_one_train_cifar100 --iteration_id=threshold_99 --pruning_threshold=0.99 --dataset_name=cifar100 --prune_whole_layers

srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/no_train_masks.py --experiment_id=threshold_exp_no_train_cifar100 --iteration_id=threshold_20 --pruning_threshold=0.20 --dataset_name=cifar100 --prune_whole_layers
srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/no_train_masks.py --experiment_id=threshold_exp_no_train_cifar100 --iteration_id=threshold_40 --pruning_threshold=0.40 --dataset_name=cifar100 --prune_whole_layers
srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/no_train_masks.py --experiment_id=threshold_exp_no_train_cifar100 --iteration_id=threshold_60 --pruning_threshold=0.60 --dataset_name=cifar100 --prune_whole_layers
srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/no_train_masks.py --experiment_id=threshold_exp_no_train_cifar100 --iteration_id=threshold_80 --pruning_threshold=0.80 --dataset_name=cifar100 --prune_whole_layers
srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/no_train_masks.py --experiment_id=threshold_exp_no_train_cifar100 --iteration_id=threshold_90 --pruning_threshold=0.90 --dataset_name=cifar100 --prune_whole_layers
srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/no_train_masks.py --experiment_id=threshold_exp_no_train_cifar100 --iteration_id=threshold_100 --pruning_threshold=1.00 --dataset_name=cifar100 --prune_whole_layers