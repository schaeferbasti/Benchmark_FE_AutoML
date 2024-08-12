#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition mlhiwidlc_gpu-rtx2080

# Define a name for your job
#SBATCH --job-name FeaEng_Pipeline

# Define the files to write the outputs of the job to.
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --error logs/%x-%A_%a.err

# Define the amount of memory required per node
#SBATCH --mem 32GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=localtmp:100

#SBATCH --propagate=NONE

# Define job array
#SBATCH --array=0-103  # Adjust based on the number of methods

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
pip install -r ../../requirements.txt
echo "Requirements installed"
# shellcheck disable=SC1068

# Set the PYTHONPATH to include the src directory
export PYTHONPATH=$PWD/src:$PYTHONPATH
echo "PYTHONPATH set to $PYTHONPATH"

# Define methods array
methods=(359944 359929 233212 359937 359950 359938 233213 359942 233211 359936 359952 359951 359949 233215 360945 167210 359943 359941 359946 360933 360932 359930 233214 359948 359931 359932 359933 359934 359939 359945 359935 317614 359940
         190411 359983 189354 189356 10090 359979 168868 190412 146818 359982 359967 359955 359960 359973 359968 359992 359959 359957 359977 7593 168757 211986 168909 189355 359964 359954 168910 359976 359969 359970 189922 359988 359984 360114 359966 211979 168911 359981 359962 360975 3945 360112 359991 359965 190392 359961 359953 359990 359980 167120 359993 190137 359958 190410 359971 168350 360113 359956 359989 359986 359975 359963 359994 359987 168784 359972 190146 359985 146820 359974 2073)
# Get the method name based on SLURM_ARRAY_TASK_ID
method=${methods[$SLURM_ARRAY_TASK_ID]}

# Run the job with the specified method
start=`date +%s`

echo "Running Method: $method"
python3 src/feature_engineering/run_feature_engineering_parallel.py --method $method

end=`date +%s`
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime"