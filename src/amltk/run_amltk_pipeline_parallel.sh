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
#SBATCH --array=0-169  # Adjust based on the number of methods

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
methods=("original1" "original5" "original14" "original15" "original16" "original17" "original18" "original21" "original22" "original23" "original24" "original27" "original28" "original29" "original31" "original35" "original36" "autofeat1" "autofeat5" "autofeat14" "autofeat15" "autofeat16" "autofeat17" "autofeat18" "autofeat21" "autofeat22" "autofeat23" "autofeat24" "autofeat27" "autofeat28" "autofeat29" "autofeat31" "autofeat35" "autofeat36" "autogluon1" "autogluon5" "autogluon14" "autogluon15" "autogluon16" "autogluon17" "autogluon18" "autogluon21" "autogluon22" "autogluon23" "autogluon24" "autogluon27" "autogluon28" "autogluon29" "autogluon31" "autogluon35" "autogluon36" "bioautoml1" "bioautoml5" "bioautoml14" "bioautoml15" "bioautoml16" "bioautoml17" "bioautoml18" "bioautoml21" "bioautoml22" "bioautoml23" "bioautoml24" "bioautoml27" "bioautoml28" "bioautoml29" "bioautoml31" "bioautoml35" "bioautoml36" "boruta1" "boruta5" "boruta14" "boruta15" "boruta16" "boruta17" "boruta18" "boruta21" "boruta22" "boruta23" "boruta24" "boruta27" "boruta28" "boruta29" "boruta31" "boruta35" "boruta36" "correlationBasedFS1" "correlationBasedFS5" "correlationBasedFS14" "correlationBasedFS15" "correlationBasedFS16" "correlationBasedFS17" "correlationBasedFS18" "correlationBasedFS21" "correlationBasedFS22" "correlationBasedFS23" "correlationBasedFS24" "correlationBasedFS27" "correlationBasedFS28" "correlationBasedFS29" "correlationBasedFS31" "correlationBasedFS35" "correlationBasedFS36" "featuretools1" "featuretools5" "featuretools14" "featuretools15" "featuretools16" "featuretools17" "featuretools18" "featuretools21" "featuretools22" "featuretools23" "featuretools24" "featuretools27" "featuretools28" "featuretools29" "featuretools31" "featuretools35" "featuretools36" "h2o1" "h2o5" "h2o14" "h2o15" "h2o16" "h2o17" "h2o18" "h2o21" "h2o22" "h2o23" "h2o24" "h2o27" "h2o28" "h2o29" "h2o31" "h2o35" "h2o36" "mljar1" "mljar5" "mljar14" "mljar15" "mljar16" "mljar17" "mljar18" "mljar21" "mljar22" "mljar23" "mljar24" "mljar27" "mljar28" "mljar29" "mljar31" "mljar35" "mljar36" "openfe1" "openfe5" "openfe14" "openfe15" "openfe16" "openfe17" "openfe18" "openfe21" "openfe22" "openfe23" "openfe24" "openfe27" "openfe28" "openfe29" "openfe31" "openfe35" "openfe36")
# Get the method name based on SLURM_ARRAY_TASK_ID
method=${methods[$SLURM_ARRAY_TASK_ID]}

# Run the job with the specified method
start=`date +%s`

echo "Running Method: $method"
python3 run_amltk_pipeline_parallel.py --method $method

end=`date +%s`
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime"