import pandas as pd
import numpy as np
import os
import subprocess
import sys
import random
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# run: Qwen

os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

num_iterations = 50
random_states = random.sample(list(range(100, 10000)), num_iterations)

# DataFrames to accumulate misclassified cases
misclassified_llm_all = pd.DataFrame(columns=["ID", "Code", "Prompt"])
misclassified_student_all = pd.DataFrame(columns=["ID", "Code", "Prompt"])

# DataFrame to store confusion matrices per iteration
confusion_matrices = []

for i, state in enumerate(random_states):
    subprocess.run([sys.executable, "scripts/dataset_sampler.py"], check=True)

    data_llm = pd.read_csv("prompting/Qwen/processed_responses.csv", header=0, names=["ID", "Prompt", "Code"])
    data_llm["label"] = 0

    data_student = pd.read_csv("CSV_files/Sampled_CodeStates.csv", header=0, names=["ID", "Code"])
    data_student["label"] = 1
    data_student["Prompt"] = ""

    data = pd.concat([data_llm, data_student], ignore_index=True)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6), max_features=1000)
    X = vectorizer.fit_transform(data["Code"]).toarray()
    y = data["label"].values

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, data.index, test_size=0.2, random_state=state)

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append({"Loopnr": i + 1, "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1]})

    misclassified_indices = np.where(y_pred != y_test)[0]
    misclassified_cases = data.loc[indices_test[misclassified_indices]].copy()

    misclassified_llm = misclassified_cases[misclassified_cases["label"] == 0][["ID", "Code", "Prompt"]]
    misclassified_student = misclassified_cases[misclassified_cases["label"] == 1][["ID", "Code", "Prompt"]]

    misclassified_llm_all = pd.concat([misclassified_llm_all, misclassified_llm], ignore_index=True)
    misclassified_student_all = pd.concat([misclassified_student_all, misclassified_student], ignore_index=True)

# Save confusion matrices
confusion_df = pd.DataFrame(confusion_matrices)
confusion_df.to_csv("ML_models/results/LightGBM_Qwen/confusion_matrices.csv", index=False)

# Save accumulated misclassifications
misclassified_llm_all.to_csv("ML_models/results/LightGBM_Qwen/misclassified_LLM_all.csv", index=False)
misclassified_student_all.to_csv("ML_models/results/LightGBM_Qwen/misclassified_Student_all.csv", index=False)

# Visualize tree
lgb.plot_tree(model, tree_index=0, figsize=(20, 8), show_info=['split_gain'])
plt.savefig("ML_models/results/LightGBM_Qwen/tree_visualization.png")
plt.close()

print("Confusion matrices and accumulated misclassified cases saved.")
