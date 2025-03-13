import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import os

# Path to your merged CSV file
csv_path = 'ML_models/code_similarity/ada_4o_students_merged.csv'

# Load CSV file
df = pd.read_csv(csv_path)

# Ensure unique IDs (useful for labeling)
df['Label'] = df['ID'].astype(str)

# Vectorize the code snippets
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Code'])

# Compute cosine similarity
similarity_matrix = cosine_similarity(X)

# Use t-SNE for dimensionality reduction
n_samples = similarity_matrix.shape[0]
perplexity = min(30, (n_samples - 1) // 3)
tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=3000, random_state=42)
X_embedded = tsne.fit_transform(similarity_matrix)

# Slightly offset overlapping points
def offset_duplicates(coords):
    unique, counts = np.unique(coords, axis=0, return_counts=True)
    offsets = np.zeros_like(coords)
    for u, count in zip(unique, counts):
        if count > 1:
            indices = np.where((coords == u).all(axis=1))[0]
            for idx, i in enumerate(indices):
                offsets[i] += (idx - count / 2) * 0.5
    return coords + offsets

X_embedded = offset_duplicates(X_embedded)

# Plotly interactive plot with edges
fig = go.Figure()

# Add edges based on similarity threshold
threshold = 0.6
for i in range(n_samples):
    for j in range(i + 1, n_samples):
        if similarity_matrix[i, j] > threshold:
            fig.add_trace(go.Scatter(x=[X_embedded[i, 0], X_embedded[j, 0]],
                                     y=[X_embedded[i, 1], X_embedded[j, 1]],
                                     mode='lines',
                                     line=dict(color='gray', width=0.5),
                                     opacity=0.3,
                                     hoverinfo='skip',
                                     showlegend=False))

# Add scatter points with hover labels
fig.add_trace(go.Scatter(
    x=X_embedded[:, 0],
    y=X_embedded[:, 1],
    mode='markers',
    marker=dict(size=10, color=df['label'], colorscale='Viridis', opacity=0.8),
    text=df['Label'],
    hoverinfo='text',
    showlegend=False
))

# Extract filename without path and extension for dynamic title
file_name = os.path.basename(csv_path).replace('.csv', '')

fig.update_layout(
    title=f'Code Similarity Visualization: {file_name}',
    xaxis_title='t-SNE Dimension 1 (reduced from cosine similarity)',
    yaxis_title='t-SNE Dimension 2 (reduced from cosine similarity)',
    width=1200, 
    height=900
)

fig.show()
