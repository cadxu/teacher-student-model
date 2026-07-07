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

import augmentations

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


def plotLoopAccuracies(md, loop_accuracies, teacher_acc):
    if not loop_accuracies:
        return

    plt.figure(figsize=(10, 6))
    loops = np.arange(1, len(loop_accuracies) + 1)
    plt.scatter(loops, loop_accuracies, s=35, label="Student OOB Accuracy")
    if len(loop_accuracies) > 1:
        slope, intercept = np.polyfit(loops, loop_accuracies, 1)
        trend = slope * loops + intercept
        plt.plot(loops, trend, color="tab:green", linewidth=2, label="Regression Line")
    plt.axhline(teacher_acc, color="tab:orange", linestyle="--", linewidth=1.5, label="Teacher OOB Accuracy")
    plt.title("OOB Accuracy by Training Loop")
    plt.xlabel("Loop")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()

    starttime = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    plt.savefig(md.name + ".loopacc." + starttime + ".png", dpi=300, bbox_inches="tight")


def make_rf(seed):
    return RandomForestClassifier(
        n_estimators=60,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=seed,
        oob_score=True,
        bootstrap=True,
    )


def build_student_training_set(md, X_pseudo, y_pseudo, seed):
    X_parts = [md.X_lab]
    y_parts = [md.y_lab]
    w_parts = [np.ones(len(md.y_lab), dtype=float)]

    if X_pseudo.size:
        X_cut, y_cut, w_cut = augmentations.compositionalCutmix(
            X_pseudo, y_pseudo, factor=2, weight=0.5, seed=seed
        )
        if X_cut.size:
            X_parts.append(X_cut)
            y_parts.append(y_cut)
            w_parts.append(w_cut)

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)
    sample_weight = np.concatenate(w_parts)
    return X_train, y_train, sample_weight


def teacherStudentLoop(md):
    """
    Implements the teacher/student self-training diagram:

      1. Initial Teacher Training  -> RF fit on labeled data only.
      2. Teacher predicts class probabilities on a weakly augmented copy
         of the unlabeled pool -> confidence threshold -> pseudo-labels
         (subset of confident samples, S < Y).
      3. Repeat (dotted portion):
           - re-augment (weak) the ORIGINAL unlabeled pool fresh each
             round (so augmentation noise never compounds)
           - Combined Labeled & Pseudo-Labeled Training Set -> train
             Student RF
           - Student RF re-predicts on the nth weakly augmented unlabeled
             matrix, thresholds confidence -> new pseudo-labels
           - Student RF's OOB accuracy is this round's "Output Accuracy"
             (no peeking at md.X_test)
      4. Final RF Model -> evaluated exactly ONCE on md.X_test.
    """
    tau = 0.7
    max_loops = 100
    min_improvement = 0.01
    md.name = "randomforest_teacher_student"

    print(f"Initial Tau: {tau}")

    teacher = make_rf(seed=42)
    teacher.fit(md.X_lab, md.y_lab)
    prev_acc = teacher.oob_score_
    print(f"Teacher OOB accuracy: {prev_acc:.4f}")
    best_model = teacher
    best_acc = prev_acc
    loop_accuracies = []

    for loop in range(max_loops):
        X_unlab_aug = augmentations.weakAugment(md.X_unlab, seed=42 + loop)
        probs = best_model.predict_proba(X_unlab_aug)
        confidences = np.max(probs, axis=1)
        pseudo_labels = np.argmax(probs, axis=1)
        mask = confidences >= tau
        X_pseudo = X_unlab_aug[mask]
        y_pseudo = pseudo_labels[mask]

        X_combined, y_combined, sample_weight = build_student_training_set(
            md, X_pseudo, y_pseudo, seed=42 + loop + 1
        )

        student = make_rf(seed=42 + loop + 1)
        student.fit(X_combined, y_combined, sample_weight=sample_weight)
        acc = student.oob_score_
        loop_accuracies.append(acc)
        delta = acc - prev_acc
        improved = acc > best_acc
        print(
            f" Loop {loop + 1}/{max_loops} | Student OOB Accuracy: {acc:.4f} "
            f"(delta={delta:+.4f}, best={best_acc:.4f}, improved={improved})"
        )

        if improved:
            best_model = student
            best_acc = acc

        prev_acc = acc
       # if loop > 0 and abs(delta) < min_improvement:
           # print(f" Converged: |delta| < {min_improvement}. Stopping.")
            #break

    plotLoopAccuracies(md, loop_accuracies, teacher.oob_score_)
    md.y_pred = best_model.predict(md.X_test)
    return md, best_model


def rfFeatureImportance(md, classifier):
    featImp = pd.DataFrame({
        "Feature": md.feature_names,
        "Importance": classifier.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print("\n=== Top Random Forest Important Features ===")
    print(featImp.head(10))

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=featImp.head(10))
    plt.title("Top 10 Important Features")

    starttime = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    plt.savefig(md.name + ".featimp." + starttime + ".png", dpi=300, bbox_inches='tight')


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

    md, rf = teacherStudentLoop(md)

    displayMetrics(md)

    printTime("Process End")


if __name__ == "__main__":
    main()