# FeatureEngineering & AutoML
This is the Version Control of the code for the Masterthesis of Bastian Schäfer, an Computer Science (AI) student at Albert-Ludwigs Universität Freiburg.

The Masterthesis is conducted on the topic of Feature Engineering in the context of AutoML.

## Get it Running:
#### Installation:
`pip install -r requirements.txt`

#### Execution:
1. Uncomment the variable `working_dir = Path("results")` to overwrite the path for the cluster execution with the correct path for local execution
2. Set the variable `rerun = True` to True if you want to rerun methods on datasets with existing results
3. Choose the variable `working_dir = Path("src/amltk/results")` accordingly, depending on you execute your code locally or in the cluster (see comments)
4. Set the variable `steps = 1`to the value of feature engineering and selection steps you want for autofeat
5. Set the variable `num_features = 50` to the desired number of generated golden features from MLJAR
6. Choose set of datasets in the line `for option in smallest_datasets:` (replace smallest_datasets by your desired set) in file src/amltk/main.py
7. Choose feature engineering methods and add them to the final dataframe in order to see the results:
   1. Get feature-engineered features by using the method from the corresponding file. Example: get_openFE_features from file src/amltk/feature_engineering/open_fe.py
   2. Get Cross-validation evaluator by calling get_cv_evaluator() method from the file src/amltk/evaluation/get_evaluator.py
   3. Optimize the pipeline and receive the history by calling e.g. history_openfe = pipeline.optimize(...)
   4. Convert history to a pandas dataframe: df_openFE = history_openFE.df()
   5. Append the dataframe with the history to one as done in line `df = pd.concat([df_original, df_sklearn, df_autofeat, df_openFE], axis=0)`
8. Execute `python3 src/amltk/pipeline/main.py`
<br>&rarr; See results in src/amltk/results/results_collection.parquet
9. Adapt the first codeblock in the src/amltk/results/analysis.ipynb file in the following way:
   1. Make sure, that the number of `max_trials` (src/amltk/main.py) still equals 10 and set the `part_size = 10` value to exactly the same value
   2. Add all labels for all the methods used in the correct order (see src/amltk/main.py `df_option = pd.concat([df_original, df_sklearn, df_autofeat, df_openFE, df_h2o, df_mljar, df_autogluon], axis=0)`)
   3. Choose the list of dataset names corresponding to the one used in the main.py file
10. Execute the file analysis.ipynb and receive all plots from the different accuracy metrics in the notebook and in the src/amltk/results/plots folder

#### Execution on MetaCluster:
1. Choose options (see above)
2. Comment the option for the working directory in case of local execution `working_dir = Path("results")`, so that the working directory for cluster execution is used (`working_dir = Path("src/amltk/results")`)
3. `sbatch run.sh`
<br>&rarr; See results in logs/AMLTK_Pipeline-_BatchJobID_.out and the src/amltk/results folder

## Explanation of the Results:
- The first 10 lines of the table are the results of 10 proposed models of the AutoML Pipeline on the original data (without Feature Engineering).
- All following lines (in packages of 10 lines each) are the results of proposed methods for AutoML with feature engineered data.

### First Insights
#### OpenFE 
- The test accuracy of the best performing model of the different splits is usually better on the original data.
- The mean of the training accuracy is usually higher over all the 10 models for the feature-engineered data.
- The mean of the validation accuracy is usually higher over all the 10 models for the feature-engineered data.
- The mean of the test accuracy is usually higher over all the 10 models for the feature-engineered data.
#### sklearn FE
- The data with the sklearn FE has a very high mean accuracy, while the std accuracy is slightly lower in comparison to the original data.
- The metric accuracy is much higher for the sklearn FE data than for the original data.
- The std accuracy is very low for both, original and FE data in all dataset splits.
- Regarding the test data, the original data outperforms the FE data.
