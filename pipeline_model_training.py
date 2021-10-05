import os 
import argparse
import pandas as pd
from azureml.core import Run
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, f1_score

# Parse the input argument 
parser = argparse.ArgumentParser()
parser.add_argument('--out_folder', type=str, dest='out_folder')
args = parser.parse_args()
output_folder = args.out_folder

# Access the data from the output of the preprocessing 
path = os.path.join(output_folder, 'preprocessed_data.csv')
prep_df = pd.read_csv(path)

X = prep_df.copy()
y = prep_df.pop('y')

# train the model 
model = RandomForestClassifier()
model.fit(X, y)
score = model.score(X, y)

y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)
f1score = f1_score(y.values, y_pred, average='macro')
prep_df['Labels'] = y_pred

# Log 
run = Run.get_context()
run.log('Confusion Matrix', cm)
run.log('Accuracy Score', score)
run.log('F1 Score', f1score)

data_prep.to_csv('./outputs/prediction_table.csv')

new_run.complete()