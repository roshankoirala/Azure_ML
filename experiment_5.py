import argparse
from azureml.core import Run, Dataset 

# Parsig argument 
parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, dest='ds_name')
args = parser.parse_args()

# Method to retriving registered data 
run = Run.get_context()
ws = run.experiment.workspace

dataset = Dataset.get_by_name(ws, args.ds_name)
data = dataset.to_pandas_dataframe()

# Log from data 
run.log('Data Size', len(data))

data.head().to_csv('./outputs/head_of_data.csv')

run.complete()