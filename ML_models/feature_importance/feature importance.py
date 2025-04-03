import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt

num_models = 10  # Number of models to average over
feature_importances_list = []

LLMs = ["Qwen", "ChatGPT4o", "ChatGPT35", "DeepSeek"]
# MLs = ["RandomForest", "SVM", "LightGBM", "NN", "XGBoost", "AdaBoost"]

MLs = ["RandomForest", "LightGBM", "AdaBoost"]

# Load each saved model and vectorizer
for LLM in LLMs:
    for ML in MLs:
        for i in range(1, num_models + 1):
            model = joblib.load(f"ML_models/feature_importance/models/{ML}_{LLM}/model_{i}.joblib")
            vectorizer = joblib.load(f"ML_models/feature_importance/models/{ML}_{LLM}/vectorizer_{i}.joblib")

            # Extract and store feature importances
            feature_importances_list.append(model.feature_importances_)

        # Compute average feature importance
        avg_importances = np.mean(feature_importances_list, axis=0)

        # Feature names from the last vectorizer (assumed consistent)
        feature_names = vectorizer.get_feature_names_out()

        # Create DataFrame for easy handling
        feature_importances_df = pd.DataFrame({
            'feature': feature_names,
            'avg_importance': avg_importances
        })

        # Sort by importance
        feature_importances_df.sort_values(by='avg_importance', ascending=False, inplace=True)

        # Plot average feature importances (top 20)
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importances_df['feature'][:20][::-1], feature_importances_df['avg_importance'][:20][::-1])
        plt.xlabel('Average Importance')
        plt.title(f"Top 20 Average Feature Importances ({ML} - {LLM})")
        plt.tight_layout()
        plt.savefig(f"ML_models/feature_importance/results/models/{ML}_{LLM}/{ML}_{LLM}_average_feature_importance.png")

