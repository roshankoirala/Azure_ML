import os
import pandas as pd
# import matplotlib.pyplot as plt 
from azureml.core import Run 

# Get experiment run context 
run = Run.get_context()

# Load the data 
df = pd.read_csv('Data/iris.csv')

n = len(df)
sample = df.head()
row = df.iloc[0].values

# Log the data 
run.log('Number of data', n)
run.log('First row', row)

os.makedirs('outputs', exist_ok=True)
sample.to_csv('outputs/head.csv', index=False, header=True)

# Complete the run 
run.complete()

