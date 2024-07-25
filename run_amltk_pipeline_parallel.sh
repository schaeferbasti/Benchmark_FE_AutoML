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
#SBATCH --array=0-99  # Adjust based on the number of methods

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
methods=("boruta1" "boruta5" "boruta14" "boruta15" "boruta16" "boruta17" "boruta18" "boruta21" "boruta22" "boruta23" "boruta24" "boruta27" "boruta28" "boruta29" "boruta31" "boruta35" "boruta36" "featurewiz1" "featurewiz5" "featurewiz14" "featurewiz15" "featurewiz16" "featurewiz17" "featurewiz18" "featurewiz21" "featurewiz22" "featurewiz23" "featurewiz24" "featurewiz27" "featurewiz28" "featurewiz29" "featurewiz31" "featurewiz35" "featurewiz36" "autogluon16" "autogluon17" "autogluon18" "autogluon27" "autogluon28" "bioautoml1" "bioautoml5" "bioautoml14" "featuretools1" "featuretools5" "featuretools14" "featuretools15" "featuretools16" "featuretools17" "featuretools18" "featuretools21" "featuretools22" "featuretools23" "featuretools24" "featuretools27" "featuretools28" "featuretools29" "featuretools31" "featuretools35" "featuretools36" "h2o1" "h2o5" "h2o14" "h2o15" "h2o16" "h2o17" "h2o18" "h2o21" "h2o22" "h2o23" "h2o24" "h2o27" "h2o28" "h2o29" "h2o31" "h2o35" "h2o36" "macfe1" "macfe5" "macfe14" "macfe15" "macfe16" "macfe17" "macfe18" "macfe21" "macfe22" "macfe23" "macfe24" "macfe27" "macfe28" "macfe29" "macfe31" "macfe35" "macfe36" "mafese1" "mafese5" "mafese14" "mafese15" "mafese16" "mafese17" "mafese18" "mafese21" "mafese22" "mafese23" "mafese24" "mafese27" "mafese28" "mafese29" "mafese31" "mafese35" "mafese36")
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