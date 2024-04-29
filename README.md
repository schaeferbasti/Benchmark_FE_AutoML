# FeatureEngineering & AutoML
This is the Version Control of the code for the Masterthesis of Bastian Schäfer, an Computer Science (AI) student at Albert-Ludwigs Universität Freiburg.

The Masterthesis is conducted on the topic of Feature Engineering in the context of AutoML.

## Get it Running:
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

## Explanation of the Results:
- The first 10 lines of the table are the results of 10 proposed models of the AutoML Pipeline on the original data (without Feature Engineering).
- The last 10 lines of the table are the results of 10 proposed models of the AutoML Pipeline with feature engineered data conducted by OpenFE.

#### First Insights
- The test accuracy of the best performing model of the different splits is usually better on the original data.
- The mean of the training accuracy is usually higher over all the 10 models for the feature-engineered data
- The mean of the validation accuracy is usually higher over all the 10 models for the feature-engineered data
- The mean of the test accuracy is usually higher over all the 10 models for the feature-engineered data

