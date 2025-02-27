#!/bin/bash -l

#SBATCH --job-name=demoohoh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=00-04:00:00       # Run time (dd-hh:mm:ss)
#SBATCH --output=/scratch/nia4240/mpp-scratch/slurm/%x/%j.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=6

export run_name=$SLURM_JOB_NAME/$SLURM_JOB_ID
export config="basic_config"   # options are "basic_config" for all or swe_only/comp_only/incomp_only/swe_and_incomp
export yaml_config="./config/mpp_avit_ti_config.yaml"

singularity exec --nv --overlay /scratch/nia4240/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate MPP
python train_basic.py --run_name $run_name --config $config --yaml_config $yaml_config 2>&1
"
