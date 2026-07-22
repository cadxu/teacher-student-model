import pandas as pd
import numpy as np
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

label_encoder = LabelEncoder()


class ModelData:
    features: np.ndarray
    labels: np.ndarray
    feature_names: list
    X_lab: np.ndarray     # labeled training inputs
    y_lab: np.ndarray     # labeled training outputs
    X_unlab: np.ndarray   # unlabeled pool
    X_test: np.ndarray    # test inputs 
    y_test: np.ndarray    # test labels 
    y_pred: np.ndarray
    name: str


def load_data(path, md):
    data = pd.read_csv(path)
    md.features = data.loc[:, data.columns != "Group"].to_numpy()
    md.feature_names = data.columns[data.columns != "Group"].tolist()
    md.labels = data['Group'].to_numpy()
    return md


def preprocess(md):

    md.labels = label_encoder.fit_transform(md.labels)

    # --- Split #1: carve off the FINAL held-out test set -------------------
    X_train_full, md.X_test, y_train_full, md.y_test = train_test_split(
        md.features, md.labels,
        test_size=0.2,
        random_state=42, stratify=md.labels
    )

    # --- Split #2: labeled vs. unlabeled pool from the remaining 80% -------
    md.X_lab, md.X_unlab, md.y_lab, _ = train_test_split(
        X_train_full, y_train_full, test_size=0.5,
        random_state=42, stratify=y_train_full
    )

    return md


def displayMetrics(md):
    print(md.name)
    print(f"Final test accuracy: {accuracy_score(md.y_test, md.y_pred):.4f}")



def make_rf(seed):
    return RandomForestClassifier(
        n_estimators=60,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=seed,
        bootstrap=True,
    )



def printTime(msg):
    starttime = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    print(msg + ": " + starttime)


def main():
    infile = ""

    if len(sys.argv) > 1:
        infile = sys.argv[1]
    else:
        print("File name ommited")
        return

    printTime("Process Start")
    md = ModelData()
    md = load_data(infile, md)
    md = preprocess(md)

    displayMetrics(md)

    printTime("Process End")


if __name__ == "__main__":
    main()