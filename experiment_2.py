import os 
import joblib 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from azureml.core import Run 

# Get the data 
df = pd.read_csv('Data/iris.csv')
X = df.copy()
y = X.pop('y')

# Train the model 
model = RandomForestClassifier()
model.fit(X, y)
score = model.score(X, y)

# Start logging 
run = Run.get_context()
run.log('Accuracy', score)

os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()