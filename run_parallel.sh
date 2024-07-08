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

# Define job array
#SBATCH --array=0-1  # Adjust based on the number of methods

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

# Define methods array
methods=("original" "autofeat") # "autogluon" "h2o" "mljar" "openfe")

# Get the method name based on SLURM_ARRAY_TASK_ID
method=${methods[$SLURM_ARRAY_TASK_ID]}

# Run the job with the specified method
start=`date +%s`

echo "Running Method: $method"
python3 src/amltk/main_parallel.py --method $method

end=`date +%s`
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime"
