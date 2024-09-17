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
#SBATCH --array=0-59  # Adjust based on the number of methods

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
export PYTHONPATH=$PWD/src:$PYTHONPATH
echo "PYTHONPATH set to $PYTHONPATH"

# Define methods array
methods=("bioautoml1" "bioautoml14" "bioautoml5" "cfs5" "featuretools1" "featuretools5" "featuretools14" "featuretools15" "featuretools16" "featuretools17" "featuretools18" "featuretools21" "featuretools22" "featuretools23" "featuretools24" "featuretools27" "featuretools28" "featuretools29" "featuretools31" "featuretools35" "featuretools36" "mafese1" "mafese5" "mafese14" "mafese15" "mafese16" "mafese17" "mafese18" "mafese21" "mafese22" "mafese23" "mafese24" "mafese27" "mafese28" "mafese29" "mafese31" "mafese35" "mafese36" "featurewiz1" "featurewiz5" "featurewiz14" "featurewiz15" "featurewiz16" "featurewiz17" "featurewiz18" "featurewiz21" "featurewiz22" "featurewiz23" "featurewiz24" "featurewiz27" "featurewiz28" "featurewiz29" "featurewiz31" "featurewiz35" "featurewiz36" "openfe29" "openfe36")
# Get the method name based on SLURM_ARRAY_TASK_ID
method=${methods[$SLURM_ARRAY_TASK_ID]}

# Run the job with the specified method
start=`date +%s`

echo "Running Method: $method"
python3 src/amltk/run_amltk_pipeline_parallel.py --method $method

end=`date +%s`
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime"