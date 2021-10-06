import os 
import numpy as np
import pandas as pd
from argparse import ArgumentParser 
from azureml.core import Run

new_run = Run.get_context()
ws = new_run.experiment.workspace

df = new_run.input_datasets['raw_data'].to_pandas_dataframe()
data_prep = df[['PL', 'SW', 'y']]

# Get the arguments from pipeline job 
parser = ArgumentParser()
parser.add_argument('--datafolder', type=str)
args = parser.parse_args()

# Create the folder if it does not exist 
os.makedirs(args.datafolder, exist_ok=True)
path = os.path.join(args.datafolder, 'output_preprocess.csv')

# Write output of preprocess as a csv file 
data_prep.to_csv(path, index=False)

# Log preprpcessing keypoints 
data_mean = data_prep.mean()
for col in data_prep.columns:
    new_run.log(col, data_mean[col])

# complete 
new_run.complete()



