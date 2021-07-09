# Deploy ML with OpenShift on IBM Cloud
# train.py
# Xavier Vasques 03/06/2021

import os
from sklearn import svm
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def train():
    # Load directory paths for persisting model

    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE = os.environ["MODEL_FILE"]
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

    # Load and split the data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)

    # Print the data
    print(X_train)
    print(X_test)

    # Train the model
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    print ("svm.SVC Model finished training")

    # Save the trained model for online inference
    dump(clf, MODEL_PATH)

if __name__ == '__main__':
    train()
