import joblib
import numpy as np
import pandas as pd
import matplotlib
from sklearn.inspection import permutation_importance

matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import os

num_models = 50  # Number of models to average over

LLMs = ["Qwen", "ChatGPT4o", "ChatGPT35", "DeepSeek"]
MLs = ["AdaBoost", "RandomForest", "LightGBM", "XGBoost"]

# Loop through each model type and LLM
for LLM in LLMs:
    for ML in MLs:
        feature_importances_list = []  # Reset for each (ML, LLM)
        
        for i in range(1, num_models + 1):
            model = joblib.load(f"ML_models/feature_importance/models/{ML}_{LLM}/model_{i}.joblib")
            vectorizer = joblib.load(f"ML_models/feature_importance/models/{ML}_{LLM}/vectorizer_{i}.joblib")
            feature_importances_list.append(model.feature_importances_)

        # Compute average feature importance
        avg_importances = np.mean(feature_importances_list, axis=0)
        feature_names = vectorizer.get_feature_names_out()

        feature_importances_df = pd.DataFrame({
            'feature': feature_names,
            'avg_importance': avg_importances
        })

        feature_importances_df.sort_values(by='avg_importance', ascending=False, inplace=True)

        # Ensure result directory exists
        output_dir = f"ML_models/feature_importance/results/models/{ML}_{LLM}"
        os.makedirs(output_dir, exist_ok=True)

        # Save CSV
        feature_importances_df.to_csv(f"{output_dir}/{ML}_{LLM}_average_feature_importance.csv", index=False)

        # Plot top 20
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importances_df['feature'][:20][::-1], feature_importances_df['avg_importance'][:20][::-1])
        plt.xlabel('Average Importance')
        plt.title(f"Top 20 Average Feature Importances ({ML} - {LLM})")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{ML}_{LLM}_average_feature_importance.png")
        plt.close()
