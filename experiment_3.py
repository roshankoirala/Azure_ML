import os 
import joblib 
import argparse
import pandas as pd 
from azureml.core import Run
from sklearn.ensemble import RandomForestClassifier

# Argument parsing 
parser = argparse.ArgumentParser()
parser.add_argument('--num_tree', type=int, dest='num_tree', default=10)
args = parser.parse_args()
n_tree = args.num_tree

# Getting data 
df = pd.read_csv('Data/iris.csv')
X = df.copy()
y = X.pop('y')

# Training model 
model = RandomForestClassifier(n_estimators=n_tree)
model.fit(X, y)
acc = model.score(X, y)

# Log 
run = Run.get_context()
run.log('Accuracy', acc)

os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()