import joblib
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LLMs = ["Qwen", "ChatGPT4o", "ChatGPT35", "DeepSeek"]

for LLM in LLMs:
    model_path = f"ML_models/feature_importance/models/SVM_{LLM}"
    feature_importances = []
    feature_names = None

    for iteration in range(1, 3):  # Adjust range if needed
        model_file = f"{model_path}/model_{iteration}.joblib"
        vectorizer_file = f"{model_path}/vectorizer_{iteration}.joblib"

        if os.path.exists(model_file) and os.path.exists(vectorizer_file):
            model = joblib.load(model_file)
            vectorizer = joblib.load(vectorizer_file)

            if feature_names is None:
                feature_names = vectorizer.get_feature_names_out()
            else:
                current_names = vectorizer.get_feature_names_out()
                if not np.array_equal(feature_names, current_names):
                    raise ValueError(f"Feature names mismatch in iteration {iteration} for {LLM}.")

            weights = np.abs(model.coef_.toarray()).flatten()
            feature_importances.append(weights)

    if not feature_importances:
        continue

    avg_importance = np.mean(feature_importances, axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'avg_importance': avg_importance.astype(float)
    })

    feature_importance_df.sort_values(by='avg_importance', ascending=False, inplace=True)

    # Plot top 20
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['feature'][:20][::-1], feature_importance_df['avg_importance'][:20][::-1])
    plt.xlabel('Average Importance')
    plt.title(f"Top 20 Average Feature Importances (SVM - {LLM})")
    plt.tight_layout()

    result_dir = f"ML_models/feature_importance/results/models/SVM_{LLM}"
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(f"{result_dir}/SVM_{LLM}_average_feature_importance.png")
    plt.close()

    # Save full CSV
    feature_importance_df.to_csv(f"{result_dir}/SVM_{LLM}_average_feature_importance.csv", index=False)
