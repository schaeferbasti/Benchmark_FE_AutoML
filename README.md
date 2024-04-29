# FeatureEngineering & AutoML
This is the Version Control of the code for the Masterthesis of Bastian Schäfer, an Computer Science (AI) student at Albert-Ludwigs Universität Freiburg.

The Masterthesis is conducted on the topic of Feature Engineering in the context of AutoML.

#### Installation:
`pip install -r requirements.txt`

#### Execution:
`python3 src/amltk/pipeline/main.py`
<br>&rarr; See results in src/amltk/pipeline/results/results.parquet

#### Execution on MetaCluster:
`sbatch scripts/meta/run.sh`
<br>&rarr; See results in logs/AMLTK_Pipeline-$date.out