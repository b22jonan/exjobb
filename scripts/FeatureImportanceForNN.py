import joblib
import numpy as np
import os

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
    feature_importance_dict = dict(zip(feature_names, avg_importance))

    # Sort and display the top N important features
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top Features for {LLM}:")
    for feature, importance in sorted_features[:20]:  # Adjust number of features
        print(f"{feature}: {importance}")
