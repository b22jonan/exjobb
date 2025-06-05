import pandas as pd 
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import time
import numpy as np
import subprocess
import sys
import joblib

LLMs = ["Qwen", "ChatGPT4o", "ChatGPT35", "DeepSeek"]

for LLM in LLMs:
    # File paths
    x_data_path = f'prompting/{LLM}/processed_responses.csv'
    failed_x_path = f'ML_models/results/XGBoost_{LLM}/Missclassified_LLM.csv'
    failed_y_path = f'ML_models/results/XGBoost_{LLM}/Missclassified_Student.csv'
    passed_x_path = f'ML_models/results/XGBoost_{LLM}/Classified_LLM.csv'
    passed_y_path = f'ML_models/results/XGBoost_{LLM}/Classified_Student.csv'
    conf_matrix_path = f'ML_models/results/XGBoost_{LLM}/confusion_matrices.csv'
    model_path = f'ML_models/feature_importance/models/XGBoost_{LLM}'
    vectorizer_path = f'ML_models/feature_importance/models/XGBoost_{LLM}'

    # Initialize lists to store failed samples and confusion matrices
    failed_x_all = pd.DataFrame()
    failed_y_all = pd.DataFrame()
    passed_x_all = pd.DataFrame()
    passed_y_all = pd.DataFrame()
    conf_matrices = []

    # Number of iterations for training
    iterations = 50

    # Start training loop
    for i in range(iterations):
        print(f"Training iteration {i + 1}...")
        
        # Generate new dataset for each iteration
        subprocess.run([sys.executable, "scripts/dataset_sampler.py"], check=True)

        # Load first dataset 
        x_data = pd.read_csv(x_data_path, header=0, names=["ID", "Prompt", "Code"])
        
        # Load second dataset
        y_data = pd.read_csv("CSV_files/Sampled_CodeStates.csv", header=0, names=["ID", "Code"])
        
        # Add labels (1 for X, 0 for Y)
        x_data['label'] = 1
        y_data['label'] = 0

        # Store ExtraField separately and drop from x_data for training
        x_extra_field = x_data[['ID', 'Prompt']]
        x_data = x_data.drop(columns=['Prompt'])

        # Combine datasets
        data = pd.concat([x_data, y_data], ignore_index=True)

        # Feature extraction using TF-IDF
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,6), max_features=1000)
        X = vectorizer.fit_transform(data['Code']).toarray()
        y = data['label']

        random_state = np.random.randint(0, 10000)  # Randomly generate random_state for each iteration
        
        # Split the data with a new random_state each time
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        model = xgb.XGBClassifier(eval_metric='logloss', random_state=random_state, max_depth=5)
        
        # Train XGBoost model
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Training time for iteration {i + 1}: {end_time - start_time:.6f} seconds")

        # Make predictions
        y_pred = model.predict(X_test)

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        conf_matrices.append([i + 1, tn, fp, fn, tp])

        # Identify misclassified entries
        misclassified = data.iloc[y_test.index][y_test != y_pred]
        
        # Ensure label column is included in the misclassified data
        misclassified = misclassified[['ID', 'Code', 'label']]

        # Separate misclassified entries for X and Y datasets
        failed_x = misclassified[misclassified['label'] == 1]
        failed_y = misclassified[misclassified['label'] == 0]
        
        # Merge back the extra field for dataset X
        failed_x = failed_x.merge(x_extra_field, on="ID", how="left")

        failed_x = failed_x[['ID', 'Code', 'Prompt']]
        failed_y = failed_y[['ID', 'Code']]

        # Append the misclassified data to the cumulative list
        failed_x_all = pd.concat([failed_x_all, failed_x], ignore_index=True)
        failed_y_all = pd.concat([failed_y_all, failed_y], ignore_index=True)

        # Identify classified samples using .iloc to prevent index errors
        classified = data.iloc[y_test.index][y_test == y_pred]

        # Ensure label column is included in the classified data
        classified = classified[['ID', 'Code', 'label']]
        
        passed_y = classified[classified['label'] == 0]
        passed_x = classified[classified['label'] == 1]

        passed_x = passed_x.merge(x_extra_field, on="ID", how="left")

        passed_x = passed_x[['ID', 'Code', 'Prompt']]
        passed_y = passed_y[['ID', 'Code']]

        passed_x_all = pd.concat([passed_x_all, passed_x], ignore_index=True)
        passed_y_all = pd.concat([passed_y_all, passed_y], ignore_index=True)

        os.makedirs(f'{model_path}', exist_ok=True)
        os.makedirs(f'{vectorizer_path}', exist_ok=True)

        joblib.dump(model, f"{model_path}/model_{i+1}.joblib")
        joblib.dump(vectorizer, f"{vectorizer_path}/vectorizer_{i+1}.joblib")

    # Save misclassified entries to CSV
    failed_x_all.to_csv(failed_x_path, index=False)
    failed_y_all.to_csv(failed_y_path, index=False)

    # Save classified entries to CSV
    passed_x_all.to_csv(passed_x_path, index=False)
    passed_y_all.to_csv(passed_y_path, index=False)

    # Save confusion matrix results to CSV
    conf_matrix_df = pd.DataFrame(conf_matrices, columns=["Loopnr", "TN", "FP", "FN", "TP"])
    conf_matrix_df.to_csv(conf_matrix_path, index=False)

    print(f'Results saved: {failed_x_path}, {failed_y_path}, {conf_matrix_path}')
