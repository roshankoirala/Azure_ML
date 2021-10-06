import os 
import pandas as pd
from azureml.core import Run
from argparse import ArgumentParser 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

# Setup azure 
parser = ArgumentParser()
parser.add_argument('--datafolder', type=str)
args = parser.parse_args()

# Access the data from the output of the preprocessing 
path = os.path.join(args.datafolder, 'output_preprocess.csv')
data_prep = pd.read_csv(path)

X = data_prep.copy()
y = data_prep.pop('y')

# train the model 
model = LogisticRegression(solver='newton-cg')
model.fit(X, y)

y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

score = model.score(X, y)
cm = confusion_matrix(y, y_pred)
f1score = f1_score(y.values, y_pred, average='macro')

data_prep['Labels'] = y_pred
data_prep['Labels_Proba'] = y_pred_proba

# Log in AzureML 
new_run = Run.get_context()
ws = new_run.experiment.workspace

new_run.log('Total Observations', len(data_prep))
new_run.log('Confusion Matrix', cm)
new_run.log('Accuracy Score', score)
new_run.log('F1 Score', f1score)

data_prep.to_csv('./outputs/prediction_table.csv')

new_run.complete()


