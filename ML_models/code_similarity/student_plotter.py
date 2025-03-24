import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pyperclip
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
import hdbscan
import seaborn as sns

# ====== Color Settings ======
# Dynamic topic colors
default_topic_colors = ['#DC143C', '#40E0D0', '#E97451', '#8B00FF', '#DAA520', '#009688', '#FF4500', '#FFD700', '#FF69B4', '#00FF00']

# ML model color mapping
ml_colors = {
    "ada": "#FF6347",         # Tomato
    "light": "#4682B4",       # Steel Blue
    "neural": "#8A2BE2",      # Blue Violet
    "rand": "#2E8B57",        # Sea Green
    "svm": "#D2691E",         # Chocolate
    "xgboost": "#20B2AA"      # Light Sea Green
}

# ====== Load CSVs ======
csv_folder = 'ML_models/code_similarity/csv_files_student/'
csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]

def extract_ml_model(filename):
    # Gets e.g. "ada" from "misclassified_student_ada_deepseek.csv"
    parts = os.path.basename(filename).split('_')
    return parts[3] if len(parts) >= 4 else 'unknown'

def load_and_merge_csvs(files):
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df['MLModel'] = extract_ml_model(file)
        df['MLColor'] = df['MLModel'].map(ml_colors).fillna('#999999')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['ID'])

df = load_and_merge_csvs(csv_files)

# ====== Preprocessing ======
def clean_code(text):
    text = text.replace('\r', '').replace('\n', ' ')
    return ' '.join(text.split())

df['Code'] = df['Code'].apply(clean_code)

# ====== Vectorization & Clustering ======
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Code'])

umap_model = umap.UMAP(n_components=2, random_state=42, spread=4, min_dist=4)
X_umap = umap_model.fit_transform(X)

# Apply HDBSCAN for topic clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
df['Topic'] = clusterer.fit_predict(X_umap)

# Assign topic colors (fallback to gray for noise)
n_topics = df['Topic'].nunique()
if n_topics > len(default_topic_colors):
    topic_colors = sns.color_palette("husl", n_topics).as_hex()
else:
    topic_colors = default_topic_colors[:n_topics]

df['TopicColor'] = df['Topic'].apply(lambda t: topic_colors[t] if t >= 0 else '#999999')

# ====== Positioning & Similarity ======
def offset_duplicates(coords, min_distance=0.5):
    seen = {}
    adjusted = coords.copy()
    for i, point in enumerate(coords):
        key = tuple(np.round(point, decimals=3))
        if key in seen:
            count = seen[key]
            adjusted[i] += np.array([min_distance * count, min_distance * count])
            seen[key] += 1
        else:
            seen[key] = 1
    return adjusted

X_umap = offset_duplicates(X_umap)
df['x'], df['y'] = X_umap[:, 0], X_umap[:, 1]

# Similarity matrix
similarity_matrix = cosine_similarity(X)

# ====== Edge Calculation ======
threshold = 0.5
edges = []
nodes_with_edges = set()
n_samples = df.shape[0]

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

# ====== Dash App ======
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Student Code Similarity (Misclassified)", style={'margin': '0px'}),
        html.P("Visualizing student code clustering & ML model distribution."),
        dcc.Checklist(
            id='filter-nodes',
            options=[
                {'label': 'Show only nodes with edges', 'value': 'only_connected'},
                {'label': 'Show only isolated nodes', 'value': 'only_isolated'},
                {'label': 'Show edges', 'value': 'show_edges'}
            ],
            value=['only_connected']
        ),
        dcc.RadioItems(
            id='color-mode',
            options=[
                {'label': 'Color by ML Model', 'value': 'ml'},
                {'label': 'Color by Topic Cluster', 'value': 'topic'}
            ],
            value='topic',
            inline=True
        )
    ], style={'textAlign': 'center', 'backgroundColor': '#f4f4f4', 'padding': '10px'}),

    html.Div([
        dcc.Graph(id='scatter-plot', style={'height': '70vh', 'width': '85vw'}, config={'scrollZoom': True})
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '75vh'}),

    html.Div(id='legend-div', style={'textAlign': 'center', 'marginTop': '20px'}),
    html.Div(id='output-div', style={'marginTop': 20, 'fontSize': 20, 'textAlign': 'center'})
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Output('legend-div', 'children'),
    Input('filter-nodes', 'value'),
    Input('color-mode', 'value')
)
def update_graph(filters, color_mode):
    show_edges = 'show_edges' in filters
    if 'only_connected' in filters:
        filtered_df = df.iloc[list(nodes_with_edges)]
    elif 'only_isolated' in filters:
        filtered_df = df[~df.index.isin(nodes_with_edges)]
    else:
        filtered_df = df

    color_column = 'MLColor' if color_mode == 'ml' else 'TopicColor'

    traces = edges if show_edges else []
    traces.append(go.Scattergl(
        x=filtered_df['x'],
        y=filtered_df['y'],
        mode='markers',
        marker=dict(size=15, color=filtered_df[color_column], opacity=0.8),
        text=filtered_df['ID'],
        customdata=filtered_df['ID'],
        hoverinfo='text'
    ))

    figure = go.Figure(data=traces, layout=go.Layout(
        title="Misclassified Code Similarity Map",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='#f4f4f4',
        plot_bgcolor='#ffffff'
    ))

    if color_mode == 'ml':
        legend = html.Div([
            html.P("ML Model Legend:", style={'fontWeight': 'bold'}),
            html.Div([
                html.Span("Adaboost", style={'color': ml_colors['ada'], 'marginRight': '20px'}),
                html.Span("LightGBM", style={'color': ml_colors['light'], 'marginRight': '20px'}),
                html.Span("Neural Net", style={'color': ml_colors['neural'], 'marginRight': '20px'}),
                html.Span("Random Forest", style={'color': ml_colors['rand'], 'marginRight': '20px'}),
                html.Span("SVM", style={'color': ml_colors['svm'], 'marginRight': '20px'}),
                html.Span("XGBoost", style={'color': ml_colors['xgboost'], 'marginRight': '20px'})
            ], style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap'})
        ])
    else:
        legend = html.Div([
            html.P("Topic Clusters (Auto-generated):", style={'fontWeight': 'bold'}),
            html.Div([
                html.Span(f"Topic {i}", style={'color': col, 'marginRight': '20px'})
                for i, col in enumerate(topic_colors)
            ],  style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap'}),
            html.P(f"Noise", style={'color': '#999999', 'marginRight': '20px'}),
        ])

    return figure, legend

@app.callback(
    Output('output-div', 'children'),
    Input('scatter-plot', 'clickData')
)
def display_click_data(clickData):
    if clickData and 'points' in clickData:
        selected_id = clickData['points'][0]['customdata']
        row = df[df['ID'] == selected_id].iloc[0]

        code_block = html.Pre(row['Code'], style={
            'backgroundColor': '#f8f8f8',
            'padding': '10px',
            'border': '1px solid #ccc',
            'overflowX': 'auto',
            'maxHeight': '300px',
            'whiteSpace': 'pre-wrap',
            'textAlign': 'left'
        })

        try:
            pyperclip.copy(str(selected_id))
            return html.Div([
                html.P(f"Copied ID: {selected_id}", style={'fontWeight': 'bold'}),
                html.P("Code Snippet:"),
                code_block
            ])
        except Exception as e:
            return f"Failed to copy ID: {str(e)}"
    
    return "Click a point to view its code and copy the ID."


if __name__ == '__main__':
    app.run_server(debug=True)
