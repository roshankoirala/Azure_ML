import joblib 
import argparse
from azureml.core import Run, Dataset, Model  

# Parsig argument 
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, dest='model_name')
parser.add_argument('--datafolder', type=str, dest='datafolder')
args = parser.parse_args()

# Method to retriving registered data 
run = Run.get_context()
ws = run.experiment.workspace

# Get the model 
model_path = Model.get_model_path(args.model_name, _workspace=ws)
model = joblib.load(model_path)

# Get data 
dataset = run.input_datasets['raw_data']
data = dataset.to_pandas_dataframe()

# Model prediction 
X = data.drop('y', axis=1)
X['pred'] = model.predict(X)

# Log from data 
run.log('Data Size', len(data))

# Create the folder if it does not exist 
os.makedirs(args.datafolder, exist_ok=True)
path = os.path.join(args.datafolder, 'iris_prediction.csv')

# Write output of preprocess as a csv file 
X.to_csv(path, index=False)

run.complete()

