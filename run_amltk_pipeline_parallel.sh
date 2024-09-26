#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition xxx

# Define a name for the job
#SBATCH --job-name AMLTK_Pipeline

# Define the files to write the outputs of the job to.
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --error logs/%x-%A_%a.err

# Define the amount of memory required per node
#SBATCH --mem 32GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=localtmp:100

#SBATCH --propagate=NONE

# Define job array
#SBATCH --array=0-191  # Adjust based on the number of methods

echo "Workingdir: $PWD"
echo "Started at $(date)"

# SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate your environment
# shellcheck disable=SC1090
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

# Define methods and datasets array
methods_datasets=("original1" "original5" "original14" "original15" "original17" "original18" "original21" "original22" "original23" "original24" "original27" "original28" "original29" "original31" "original35" "original36" "autofeat1" "autofeat5" "autofeat14" "autofeat15" "autofeat17" "autofeat18" "autofeat21" "autofeat22" "autofeat23" "autofeat24" "autofeat27" "autofeat28" "autofeat29" "autofeat31" "autofeat35" "autofeat36" "autogluon1" "autogluon5" "autogluon14" "autogluon15" "autogluon17" "autogluon18" "autogluon21" "autogluon22" "autogluon23" "autogluon24" "autogluon27" "autogluon28" "autogluon29" "autogluon31" "autogluon35" "autogluon36" "bioautoml1" "bioautoml5" "bioautoml14" "bioautoml15" "bioautoml17" "bioautoml18" "bioautoml21" "bioautoml22" "bioautoml23" "bioautoml24" "bioautoml27" "bioautoml28" "bioautoml29" "bioautoml31" "bioautoml35" "bioautoml36" "boruta1" "boruta5" "boruta14" "boruta15" "boruta17" "boruta18" "boruta21" "boruta22" "boruta23" "boruta24" "boruta27" "boruta28" "boruta29" "boruta31" "boruta35" "boruta36" "cfs1" "cfs5" "cfs14" "cfs15" "cfs17" "cfs18" "cfs21" "cfs22" "cfs23" "cfs24" "cfs27" "cfs28" "cfs29" "cfs31" "cfs35" "cfs36" "featurewiz1" "featurewiz5" "featurewiz14" "featurewiz15" "featurewiz17" "featurewiz18" "featurewiz21" "featurewiz22" "featurewiz23" "featurewiz24" "featurewiz27" "featurewiz28" "featurewiz29" "featurewiz31" "featurewiz35" "featurewiz36" "h2o1" "h2o5" "h2o14" "h2o15" "h2o17" "h2o18" "h2o21" "h2o22" "h2o23" "h2o24" "h2o27" "h2o28" "h2o29" "h2o31" "h2o35" "h2o36" "macfe1" "macfe5" "macfe14" "macfe15" "macfe17" "macfe18" "macfe21" "macfe22" "macfe23" "macfe24" "macfe27" "macfe28" "macfe29" "macfe31" "macfe35" "macfe36" "mafese1" "mafese5" "mafese14" "mafese15" "mafese17" "mafese18" "mafese21" "mafese22" "mafese23" "mafese24" "mafese27" "mafese28" "mafese29" "mafese31" "mafese35" "mafese36" "mljar1" "mljar5" "mljar14" "mljar15" "mljar17" "mljar18" "mljar21" "mljar22" "mljar23" "mljar24" "mljar27" "mljar28" "mljar29" "mljar31" "mljar35" "mljar36" "openfe1" "openfe5" "openfe14" "openfe15" "openfe17" "openfe18" "openfe21" "openfe22" "openfe23" "openfe24" "openfe27" "openfe28" "openfe29" "openfe31" "openfe35" "openfe36")
# Get the method name and dataset number based on SLURM_ARRAY_TASK_ID
method_dataset=${methods_datasets[$SLURM_ARRAY_TASK_ID]}

# Run the job with the specified method
# shellcheck disable=SC2006
start=`date +%s`

echo "Running Method: $method_dataset"
python3 src/amltk/run_amltk_pipeline_parallel.py --method_dataset "$method_dataset"

# shellcheck disable=SC2006
end=`date +%s`
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime"