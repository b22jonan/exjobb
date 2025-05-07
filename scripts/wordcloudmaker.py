import os
import pandas as pd
from collections import defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt

LLMs = ["Qwen", "ChatGPT4o", "ChatGPT35", "DeepSeek"]
MLs = ["AdaBoost", "RandomForest", "LightGBM", "XGBoost", "SVM", "NN"]
base_path = "ML_models/feature_importance/results/models"

for LLM in LLMs:
    print(f"Processing LLM: {LLM}")
    feature_importance_total = defaultdict(float)

    for ML in MLs:
        path = f"{base_path}/{ML}_{LLM}/{ML}_{LLM}_average_feature_importance.csv"
        if not os.path.exists(path):
            print(f"Warning: Missing file {path}")
            continue

        df = pd.read_csv(path)
        df['normalized_importance'] = df['avg_importance'] / df['avg_importance'].sum()
        
        for _, row in df.iterrows():
            feature_importance_total[row['feature']] += row['normalized_importance']

    if not feature_importance_total:
        print(f"No data available for {LLM}, skipping.")
        continue

    agg_df = pd.DataFrame(feature_importance_total.items(), columns=['feature', 'total_importance'])
    top_features = agg_df.sort_values(by='total_importance', ascending=False).head(50)
    feature_dict = dict(zip(top_features['feature'], top_features['total_importance']))

    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        collocations=False,
        prefer_horizontal=1.0,
        min_font_size=10
    ).generate_from_frequencies(feature_dict)
    
    plt.figure(figsize=(14, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Top 50 Features for LLM: {LLM}", fontsize=20)
    
    save_dir = f"ML_models/feature_importance/results/overlap/{LLM}"
    os.makedirs(save_dir, exist_ok=True)
    
    output_path = os.path.join(save_dir, f"{LLM}_wordcloud.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved improved word cloud for {LLM} to {output_path}")
