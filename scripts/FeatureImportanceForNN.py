import joblib
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt

LLMs = ["Qwen", "ChatGPT4o", "ChatGPT35", "DeepSeek"]

for LLM in LLMs:
    model_path = f"ML_models/feature_importance/models/NN_{LLM}"
    vectorizer_path = f"ML_models/feature_importance/models/NN_{LLM}/vectorizer_1.joblib"

    vectorizer = joblib.load(vectorizer_path)
    feature_names = vectorizer.get_feature_names_out()
    
    feature_importances = []

    for iteration in range(1, 3):  # Adjust range if you have more iterations
        model_file = f"{model_path}/model_{iteration}.joblib"
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            weights = np.abs(model.coefs_[0]).mean(axis=1)  # Importance from first layer
            feature_importances.append(weights)

    # Average over all iterations
    avg_importance = np.mean(feature_importances, axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'avg_importance': avg_importance
    })

    # Sort by importance
    feature_importance_df.sort_values(by='avg_importance', ascending=False, inplace=True)

    # Plot average feature importances (top 20)
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['feature'][:20][::-1], feature_importance_df['avg_importance'][:20][::-1])
    plt.xlabel('Average Importance')
    plt.title(f"Top 20 Average Feature Importances (NN - {LLM})")
    plt.tight_layout()
    save_path = f"ML_models/feature_importance/results/models/NN_{LLM}/NN_{LLM}_average_feature_importance.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
