import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pyperclip
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Directory containing CSV files
csv_folder = 'ML_models/code_similarity/csv_files/'
csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]

# Load and merge all CSV files
def load_and_merge_csvs(csv_files):
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)  # Track source file
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['ID'])  # Remove duplicate IDs

# Load all data
df = load_and_merge_csvs(csv_files)

# Vectorize the code snippets
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Code'])

# **Optimize with PCA before t-SNE**
pca = TruncatedSVD(n_components=50)  # Reduce dimensionality to 50 before t-SNE
X_reduced = pca.fit_transform(X)

# Compute cosine similarity
similarity_matrix = cosine_similarity(X_reduced)

# Use t-SNE for dimensionality reduction
n_samples = similarity_matrix.shape[0]
perplexity = min(30, (n_samples - 1) // 3)
tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
X_embedded = tsne.fit_transform(X_reduced)

# Offset overlapping points in a consistent direction
def offset_duplicates(coords, min_distance=0.5):
    seen = {}
    adjusted_coords = coords.copy()
    offset_step = min_distance  # Set step size for offset
    for i, point in enumerate(coords):
        key = tuple(np.round(point, decimals=3))
        if key in seen:
            count = seen[key]
            adjusted_coords[i] += np.array([offset_step * count, offset_step * count])
            seen[key] += 1
        else:
            seen[key] = 1
    return adjusted_coords

X_embedded = offset_duplicates(X_embedded)
df['x'] = X_embedded[:, 0]
df['y'] = X_embedded[:, 1]

# Set similarity threshold for edges (Reduce for performance)
threshold = 0.5  # Lowered for more edges
edges = []
nodes_with_edges = set()
df = df.reset_index(drop=True)  # Reset index after filtering out duplicates

for i in range(n_samples):
    for j in range(i + 1, n_samples):
        if similarity_matrix[i, j] > threshold:
            edges.append(
                go.Scattergl(
                    x=[df.iloc[i]['x'], df.iloc[j]['x']],
                    y=[df.iloc[i]['y'], df.iloc[j]['y']],
                    mode='lines',
                    line=dict(color='gray', width=0.3),
                    opacity=0.2,
                    hoverinfo='skip',
                    showlegend=False
                )
            )
            nodes_with_edges.add(i)
            nodes_with_edges.add(j)

# Define color mapping
def num_to_color(num):
    color_map = {
        1: '#DC143C',  # Crimson (Copy Paste)
        2: '#40E0D0',  # Turquoise (Perfect)
        3: '#E97451',  # Burnt Sienna (Memetic Proxy)
        4: '#8B00FF',  # Electric Violet (Meta)
        5: '#DAA520',  # Goldenrod (Restraints)
        6: '#009688',  # Teal (Translation)
    }
    return color_map.get(num, '#999999')  # Default: Neutral Gray if not found

# Dash App
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Div(
        style={'backgroundColor': '#f4f4f4', 'padding': '10px', 'textAlign': 'center'},
        children=[
            html.H1("Code Similarity Visualization", style={'margin': '0px'}),
            html.P("Merging all datasets for visualization."),
            dcc.Checklist(
                id='filter-nodes',
                options=[
                    {'label': 'Show only nodes with similarity edges', 'value': 'only_connected'},
                    {'label': 'Show only isolated nodes', 'value': 'only_isolated'},
                    {'label': 'Show edges', 'value': 'show_edges'}
                ],
                value=['only_connected']
            )
        ]
    ),
    html.Div(
        style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '75vh'},
        children=[
            html.Div(
                style={'border': '2px solid black', 'padding': '10px', 'backgroundColor': 'white'},
                children=[
                    dcc.Graph(
                        id='scatter-plot',
                        style={'height': '70vh', 'width': '85vw'},
                        config={'scrollZoom': True, 'displayModeBar': True}
                    )
                ]
            )
        ]
    ),
    html.Div(
        style={'textAlign': 'center', 'marginTop': '20px'},
        children=[
            html.P("Legend:", style={'fontWeight': 'bold'}),
            html.Ul([
                html.Li("Copy Paste - Crimson", style={'color': '#DC143C'}),
                html.Li("Perfect - Turquoise", style={'color': '#40E0D0'}),
                html.Li("Memetic Proxy - Burnt Sienna", style={'color': '#E97451'}),
                html.Li("Meta - Electric Violet", style={'color': '#8B00FF'}),
                html.Li("Restraints - Goldenrod", style={'color': '#DAA520'}),
                html.Li("Translation - Teal", style={'color': '#009688'})
            ])
        ]
    ),
    html.Div(id='output-div', style={'marginTop': 20, 'fontSize': 20, 'textAlign': 'center'})
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('filter-nodes', 'value')
)
def update_graph(filter_value):
    show_edges = 'show_edges' in filter_value

    # **Fix for Node Filtering**
    if 'only_connected' in filter_value:
        filtered_df = df.iloc[list(nodes_with_edges)]
    elif 'only_isolated' in filter_value:
        filtered_df = df[~df.index.isin(nodes_with_edges)]
    else:
        filtered_df = df

    figure = go.Figure(
        data=(edges if show_edges else []) + [
            go.Scattergl(
                x=filtered_df['x'],
                y=filtered_df['y'],
                mode='markers',
                marker=dict(size=15, color=[num_to_color(val) for val in filtered_df['PromptType']], opacity=0.8),
                text=filtered_df['ID'],
                customdata=filtered_df['ID'],
                hoverinfo='text'
            )
        ],
        layout=go.Layout(
            title="Code Similarity Visualization (Filtered View)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='#f4f4f4',
            plot_bgcolor='#ffffff'
        )
    )
    return figure

@app.callback(
    Output('output-div', 'children'),
    Input('scatter-plot', 'clickData')
)
def display_click_data(clickData):
    if clickData and 'points' in clickData:
        selected_id = clickData['points'][0]['customdata']
        pyperclip.copy(str(selected_id))
        return f"Copied ID: {selected_id}"
    return "Click a point to copy its ID."

if __name__ == '__main__':
    app.run_server(debug=True)
