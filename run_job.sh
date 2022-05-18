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
python3 -m pip install --user transformers
python3 -m pip install --user datasets
python3 -m pip install --user sklearn
python3 -m pip install --user tensorboardX
python3 -m pip install --user matplotlib



## Run code
srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/find_head_mask_smart_deit.py --iteration_id=deit_no_normalization --dont_normalize_global_importance --dont_normalize_importance_by_layer
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/find_head_mask_smart.py --iteration_id=iterative_test_1
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/find_head_mask_1.py --output_dir=saved_numpys_cifar100_1_1 --iteration_number=1
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/find_head_mask_0.py --iteration_id=attempt7_rerun
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/mask_head_vit_2_iter.py --output_dir=saved_numpys_mnist_2_iter --dataset_name=mnist
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/mask_head_vit_2_iter.py --output_dir=saved_numpys_cifar10_2_iter --dataset_name=cifar10
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/main.py --head_mask=saved_numpys/head_mask_5.npy --mask_heads
#srun python3 /rwthfs/rz/cluster/home/rs062004/tmp/pycharm_project_109/full_train_with_masks.py --mask_heads --model_name=starting_checkpoint_cifar100_iterative_test_1 --numpy_dir=saved_numpys_iterative_test_1

