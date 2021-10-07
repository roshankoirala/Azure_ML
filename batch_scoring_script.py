# import io
# import numpy as np
# from sklearn.linear_model import LogisticRegression

import pickle
import argparse
from azureml.core.model import Model
from azureml_user.parallel_run import EntryScript

def init():
    global model

    logger = EntryScript().logger
    logger.info("init() is called.")

    parser = argparse.ArgumentParser(description='Iris model serving')
    parser.add_argument('--model_name', dest='model_name', required=True)
    args, unknown_args = parser.parse_known_args()

    model_path = Model.get_model_path(args.model_name)
    with open(model_path, 'rb') as model_file:
        iris_model = pickle.load(model_file)


def run(input_data):
    
    logger = EntryScript().logger
    logger.info("run() is called with: {}.".format(input_data))

    # make inference
    pred = iris_model.predict(input_data)
    result['variety'] = pred
    
    return result