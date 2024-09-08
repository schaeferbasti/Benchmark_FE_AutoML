# FeatureEngineering & AutoML
This is the Version Control of the code for my master project "Benchmarking Feature Engineering for AutoML": My name is Bastian Schäfer and I am a Computer Science (AI) student at Albert-Ludwigs Universität Freiburg doing my project with the ML Lab (Prof. Frank Hutter) supervised by Lennart Purucker.

The master project is conducted on the topic of Feature Engineering in the context of AutoML. The goal is to investigate feature engineering in the AutoML context and show the benefits of (not) using it by implementing a benchmark.


## Details on the Benchmark
There are two different implementations of the benchmark, a simpler one and a more sophisticated one.
The simpler one is the AMLTK pipeline, which is using a LGBM classifier and performs feature engineering with the given methods on 17 datasets and uses as many Trials as possible in one hour to find the best configuration for the given data using a RandomSearch Optimizer. The evaluation is conducted by using the ROC AUC metric.
The more sophisticated one is based on Autogluon, we are using 4 hours of time (32GB, 8 CPUs, 104 datasets) for feature engineering and AutoML together, so that all methods have to work with the same amount of time. Autogluon automatically searches for the best classifier by itself. We are using different metrics for the final evaluation depending on the task.


## Code Structure
The code is structured in 4 parts, in the src folder one can find an amltk directory, an autogluon directory, a datasets directory and a feature_engineering directory.
The amltk and autogluon folders contain the code of the respective pipelines, that can be used by running the corresponding run_xxx.py file.
In the datasets folder, there is the code for retrieving datasets in the Datasets.py file and the code for the splits in the Splits file. There is also a directory containing all feature engineered datasets as .parquet files.
In the feature_engineering folder, there is the code for all the tested feature engineering methods as far as they are open-source and there is a file for executing all feature engineering methods on the AMLB datasets and save the results to a file.

For the amltk, autogluon and feature engineering, there are python files for the normal execution (e.g. run_amltk_pipeline.py) and a parallelized execution (e.g. run_amltk_pipeline_parallel.py) in the respective folders. The repository also provides batch files (e.g. run_amltk_pipeline_parallel.sh) in the default directory to execute the run files.

## Get it Running:
#### Install Dependencies:
`pip install -r requirements.txt`

#### Local Execution of AMLTK:
1. Choose the variable `working_dir = Path("src/amltk/results")` accordingly, depending on you execute your code locally or in the cluster (see comments)
2. Set the variable `rerun = True` to True if you want to rerun methods on datasets with existing results
3. Set the variable `debugging = False` to True if you want to raise Trial Exceptions
4. Choose set of datasets in the line `for option in all_datasets:` (replace all_datasets by your desired set) in file src/amltk/main.py
5. Choose feature engineering methods and add them to the final dataframe in order to see the results:
   1. Get feature-engineered features by using the method from the corresponding file. Example: `get_openFE_features` from file src/amltk/feature_engineering/open_fe.py
   2. Get Cross-validation evaluator by calling `get_cv_evaluator()` method from the file src/amltk/evaluation/get_evaluator.py
   3. Optimize the pipeline and receive the history by calling e.g. `history_openfe = pipeline.optimize(...)`
   4. Convert history to a pandas dataframe: `df_openFE = history_openFE.df()`
   5. Append the dataframe with the history to one as done in line `df = pd.concat([df_original, df_sklearn, df_autofeat, df_openFE], axis=0)`
6. Execute `python3 src/amltk/run_amltk_pipeline.py`
7. In case of parallel execution use the batch file `src/amltk/run_amltk_pipeline_parallel.sh`
<br>&rarr; See all results in src/amltk/results/results_collection.parquet 
8. Adapt the first codeblock in the src/amltk/results/analysis.ipynb file in the following way:
   1. Make sure, that the number of `max_trials` (src/amltk/run_amltk_pipeline.py) still equals 10 and set the `part_size = 10` value to exactly the same value
   2. Add all labels for all the methods used in the correct order (see src/amltk/main.py `df_option = pd.concat([df_original, df_sklearn, df_autofeat, df_openFE, df_h2o, df_mljar, df_autogluon], axis=0)`)
   3. Choose the list of dataset names corresponding to the one used in the main.py file 
9. Execute the file analysis.ipynb and receive all plots from the different accuracy metrics in the notebook and in the src/amltk/results/plots folder
10. Execute the file tabular_analysis.ipynb for a tabular comparison

#### Local Execution of Autogluon:
1. Change the variables `datasets` in case of need for other datasets and the `methods` variable in case of usage of other feature engineering methods
2. Execute `run_autogluon.py`


#### Local Execution of Feature Engineering:
1. Change the variables `amlb_task_ids` in case of need for other datasets and the `feature_engineering_methods` in case of other feature engineering methods
2. Execute `run_feature_engineering.py`

#### Execution on MetaCluster:
1. Choose options (see above)
2. Run according batch file (that has the same name) by submitting a job with `sbatch run.sh`
<br>&rarr; See all results in logs/AMLTK_Pipeline-_BatchJobID_.out and in the file src/amltk/results/results_collection.parquet
3. Retrieve results by compressing the results folder (e.g. `tar -czvf src/amltk/results.tar.gz src/amltk/results/`) and downloading the compressed file via `scp yourusername@kislogin1.rz.ki.privat:FE_AutoML/src/amltk/results.tar.gz Downloads`

