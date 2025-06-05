import pandas as pd
import itertools
import os
import matplotlib.pyplot as plt

LLMs = ["Qwen", "ChatGPT4o", "ChatGPT35", "DeepSeek"]
MLs = ["AdaBoost", "RandomForest", "LightGBM", "XGBoost", "SVM", "NN"]

top_n = 200

for LLM in LLMs:
    print(f"\nAnalyzing feature overlap for LLM: {LLM}")

    importance_dfs = {}
    for ML in MLs:
        path = f"ML_models/feature_importance/results/models/{ML}_{LLM}/{ML}_{LLM}_average_feature_importance.csv"
        if not os.path.exists(path):
            print(f"Warning: Missing file {path}")
            continue
        
        df = pd.read_csv(path)

        df['normalized_importance'] = df['avg_importance'] / df['avg_importance'].sum()

        top_features = df.sort_values(by='normalized_importance', ascending=False).head(top_n)['feature'].tolist()
        importance_dfs[ML] = set(top_features)

    used_MLs = list(importance_dfs.keys())
    overlap_matrix = pd.DataFrame(index=used_MLs, columns=used_MLs)

    for ml1, ml2 in itertools.product(used_MLs, repeat=2):
        common_features = importance_dfs[ml1].intersection(importance_dfs[ml2])
        overlap_ratio = len(common_features) / top_n
        overlap_matrix.loc[ml1, ml2] = overlap_ratio

    output_dir = f"ML_models/feature_importance/results/overlap/{LLM}"
    os.makedirs(output_dir, exist_ok=True)
    overlap_matrix.to_csv(f"{output_dir}/overlap_matrix_top{top_n}.csv")

    plt.figure(figsize=(8, 6))
    plt.imshow(overlap_matrix.astype(float), cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(len(used_MLs)), used_MLs, rotation=45)
    plt.yticks(range(len(used_MLs)), used_MLs)
    plt.colorbar(label="Top Feature Overlap Ratio")
    plt.title(f"Feature Overlap Matrix (Top {top_n}) - {LLM}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overlap_matrix_top{top_n}.png")
    plt.close()
