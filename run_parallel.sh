#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition mlhiwidlc_gpu-rtx2080

# Define a name for your job
#SBATCH --job-name AMLTK_Pipeline

# Define the files to write the outputs of the job to.
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --error logs/%x-%A_%a.err

# Define the amount of memory required per node
#SBATCH --mem 48GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=localtmp:100

#SBATCH --propagate=NONE

# Define methods, datasets, and folds
methods=("original" "autofeat" "autogluon" "bioautoml" "boruta" "correlationBasedFS" "featuretools" "h2o" "mljar" "openfe")
datasets=(1 5 14 15 16 17 18 21 22 23 24 27 28 29 31 35 36)
folds=$(seq 0 9)

# Calculate total number of jobs (methods * datasets * folds)
total_jobs=$((${#methods[@]} * ${#datasets[@]} * ${#folds[@]}))

# Define job array
#SBATCH --array=0-$(($total_jobs-1))

echo "Workingdir: $PWD"
echo "Started at $(date)"

# SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate your environment
source ~/miniconda3/bin/activate
conda activate amltk_env
echo "conda amltk_env activated"

# Install requirements
python3 -m pip install --upgrade pip
pip install -r requirements.txt
echo "Requirements installed"
# shellcheck disable=SC1068

# Set the PYTHONPATH to include the src directory
export PYTHONPATH=$PWD/src/amltk:$PYTHONPATH
echo "PYTHONPATH set to $PYTHONPATH"

# Calculate method, dataset, and fold based on SLURM_ARRAY_TASK_ID
method_index=$(($SLURM_ARRAY_TASK_ID % ${#methods[@]}))
dataset_index=$((($SLURM_ARRAY_TASK_ID / ${#methods[@]}) % ${#datasets[@]}))
fold_index=$(($SLURM_ARRAY_TASK_ID / (${#methods[@]} * ${#datasets[@]})))

method=${methods[$method_index]}
dataset=${datasets[$dataset_index]}
fold=$fold_index

echo "Running Method: $method, Dataset: $dataset, Fold: $fold"

# Run the job with the specified method
start=`date +%s`

python3 src/amltk/main_parallel.py --method $method --dataset $dataset --fold $fold

end=`date +%s`
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime"
