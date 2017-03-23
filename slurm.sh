#!/bin/bash -l


#SBATCH
#SBATCH --job-name=keras-fcn
#SBATCH --time=24:0:0
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
# number of cpus (threads) per task (process)
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=end
#SBATCH --mail-user=ahundt1@jhu.edu
#SBATCH --output=logs/output_keras-fcn.log
#SBATCH --error=logs/error_keras-fcn.log


#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

source /home-1/ahundt1@jhu.edu/.robotics_setup
source /home-1/ahundt1@jhu.edu/src/robotics_setup/marcc-config.sh

export PYTHONPATH=$PYTHONPATH:/cm/shared/apps/Intel/python/2.7.10b/lib/python2.7/site-packages:/home-1/ahundt1@jhu.edu/.local/lib/python2.7/site-packages/:/home-1/ahundt1@jhu.edu/src/tf-image-segmentation:/home-1/ahundt1@jhu.edu/src/tensorflow_slurm_manager:/home-1/ahundt1@jhu.edu/src/tf-image-segmentation:/home-1/ahundt1@jhu.edu/src/Keras-FCN

cd /home-1/ahundt1@jhu.edu/src/Keras-FCN
./train.py
./evaluate.py