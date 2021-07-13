#!/usr/bin/python3
# api.py
# Xavier Vasques 03/06/2021

import os
from sklearn import svm
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from joblib import load

from flask import Flask

# Set environnment variables
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Loading model
print("Loading model from: {}".format(MODEL_PATH))
inference = load(MODEL_PATH)

# Creation of the Flask app
app = Flask(__name__)

# API 
# Flask route so that we can serve HTTP traffic on that route
@app.route('/',methods=['POST', 'GET'])
# Return predictions of inference using Iris Test Data
def prediction():

    # Load and split the data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)

    # Classification score
    clf = load(MODEL_PATH)
    score = clf.score(X_test, y_test)

    return {'score': score}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080) # Launch built-in we server and run this Flask webapp



