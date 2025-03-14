import pandas as pd
import numpy as np
import os
import subprocess
import sys
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import graphviz

# run: Qwen

# Ensure Graphviz is in PATH (adjust as needed)
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

num_iterations = 50
random_states = random.sample(list(range(100, 10000)), num_iterations)

# DataFrames to accumulate misclassified cases
misclassified_llm_all = pd.DataFrame(columns=["ID", "code", "prompt", "label"])
misclassified_student_all = pd.DataFrame(columns=["ID", "code", "prompt", "label"])

# DataFrame to store confusion matrices per iteration
confusion_matrices = []

for i, state in enumerate(random_states):
    subprocess.run([sys.executable, "scripts/dataset_sampler.py"], check=True)

    data_llm = pd.read_csv("prompting/Qwen/processed_responses.csv", header=0, names=["ID", "prompt", "code"])
    data_llm["label"] = 0

    data_student = pd.read_csv("CSV_files/Sampled_CodeStates.csv", header=0, names=["ID", "code"])
    data_student["label"] = 1
    data_student["prompt"] = ""

    data = pd.concat([data_llm, data_student], ignore_index=True)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6), max_features=1000)
    X = vectorizer.fit_transform(data["code"]).toarray()
    y = data["label"].values

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, data.index, test_size=0.2, random_state=state)

    base_model = DecisionTreeClassifier(max_depth=3)
    model = AdaBoostClassifier(base_model, n_estimators=50, learning_rate=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append({"Loopnr": i + 1, "tn": cm[0,0], "fp": cm[0,1], "fn": cm[1,0], "tp": cm[1,1]})

    misclassified_indices = np.where(y_pred != y_test)[0]
    misclassified_cases = data.loc[indices_test[misclassified_indices]].copy()

    misclassified_llm = misclassified_cases[misclassified_cases["label"] == 0][["ID", "code", "prompt", "label"]]
    misclassified_student = misclassified_cases[misclassified_cases["label"] == 1][["ID", "code", "prompt", "label"]]

    misclassified_llm_all = pd.concat([misclassified_llm_all, misclassified_llm], ignore_index=True)
    misclassified_student_all = pd.concat([misclassified_student_all, misclassified_student], ignore_index=True)
    print(f"Iteration {i + 1} completed.")

# Save confusion matrices
confusion_matrices_df = pd.DataFrame(confusion_matrices)
confusion_matrices_df.to_csv("ML_models/results/AdaBoost_Qwen/confusion_matrices.csv", index=False)

# Save accumulated misclassified cases
misclassified_llm_all.to_csv("ML_models/results/AdaBoost_Qwen/misclassified_LLM_all.csv", index=False)
misclassified_student_all.to_csv("ML_models/results/AdaBoost_Qwen/misclassified_Student_all.csv", index=False)

# Visualize an individual decision tree from the final model
if hasattr(model, "estimators_") and len(model.estimators_) > 0:
    tree = model.estimators_[0]
    dot_data = export_graphviz(
        tree,
        feature_names=vectorizer.get_feature_names_out(),
        class_names=["LLM", "Student"],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render("ML_models/results/AdaBoost_Qwen/tree_visualization")

print("Confusion matrices and accumulated misclassified cases saved.")
