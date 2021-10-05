import os 
import argparse
from azureml.core import Run 

# Parse the input argument 
parser = argparse.ArgumentParser()
parser.add_argument('--out_folder', type=str, dest='out_folder')
args = parser.parse_args()
output_folder = args.out_folder

# Get the raw data 
run = Run.get_context()
raw_df = run.input_df['raw_data'].to_pandas_dataframe()

# Do preprocessing 
# Insert pickle preprocessing files here???
select_cols = ['PL', 'SW', 'y']
prep_df = raw_df[select_cols]

# Save the preprocessed data 
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'preprocessed_data.csv')
prep_df.to_csv(output_path)

# Log head of the data, shape of the data, column names, summary statistics table 
# what else should be logged??? 
run.log('Shape of data', prep_df.shape())

# Complete the run 
run.complete()