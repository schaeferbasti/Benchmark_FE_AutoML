# FeatureEngineering & AutoML
This is the Version Control of the code for the Masterthesis of Bastian Schäfer, an Computer Science (AI) student at Albert-Ludwigs Universität Freiburg.

The Masterthesis is conducted on the topic of Feature Engineering in the context of AutoML.

#### Installation:
`pip install -r requirements.txt`

#### Execution:
1. Choose option for dataset in the line `X_original, X_test_original, y, y_test = get_dataset(option=2, openml_task_id=openml_task_id, outer_fold_number=outer_fold_number)` in file src/amltk/pipeline/main.py
- Option 1: california-housing dataset from OpenFE example
- Option 2: cylinder-bands dataset from OpenFE benchmark
- Option 3: balance-scale dataset from OpenFE benchmark (not working)
- Option 4: black-friday dataset from AMLB (long execution time)

2. `python3 src/amltk/pipeline/main.py`
<br>&rarr; See results in src/amltk/pipeline/results/results.parquet

#### Execution on MetaCluster:
1. Choose option (see above)
2. `sbatch scripts/meta/run.sh`
<br>&rarr; See results in logs/AMLTK_Pipeline-_BatchJobID_.out