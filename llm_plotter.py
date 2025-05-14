import os
import pandas as pd
import numpy as np
import seaborn as sns
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
import hdbscan
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
from dash import ClientsideFunction

# Colors and mappings
topic_colors = ['#DC143C', '#40E0D0', '#E97451', '#8B00FF', '#DAA520', '#009688', '#FF4500', '#FFD700', '#FF69B4', '#00FF00']
llm_colors = {"ChatGPT4o": "#FF4500", "ChatGPT35": "#1E90FF", "DeepSeek": "#32CD32", "Qwen": "#DFB700"}
class_colors = {
    "classified_LLM": "#666666",
    "classified_Student": "#FF4500",
    "misclassified_Student": "#ff00ff",
    "misclassified_LLM": "#000000",
}
def num_to_color(num):
    return {
        1: '#DC143C', 2: '#40E0D0', 3: '#E97451',
        4: '#8B00FF', 5: '#DAA520', 6: '#009688'
    }.get(num, '#999999')

csv_dirs = ['ML_models/code_similarity/csv_files_llm_not_in_use/', 'ML_models/code_similarity/csv_files_student_not_in_use/']
csv_files = [os.path.join(d, f) for d in csv_dirs for f in os.listdir(d) if f.endswith('.csv')]

# Utility functions
def extract_llm_from_filename(fname): return fname.split('_')[-1].replace('.csv', '')
def extract_class_from_filename(fname): return '_'.join(fname.split('_')[1:-2])

def load_and_merge_csvs(files):
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        if 'prompt' in df.columns: df.drop(columns='prompt', inplace=True)
        df['source_file'] = os.path.basename(file)
        df['LLM'] = extract_llm_from_filename(file)
        df['Class'] = extract_class_from_filename(file)
        df['LLMColor'] = df['LLM'].map(llm_colors)
        df['ClassColor'] = df['Class'].map(class_colors)
        df['PromptColor'] = df['PromptType'].map(num_to_color)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['ID'])

def generate_edges(df, sim_matrix, threshold=0.5, cluster_threshold=5):
    raw_edges, optimized_edges = [], []
    nodes_with_edges = set()
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if sim_matrix[i, j] > threshold:
                raw_edges.append((i, j))
                nodes_with_edges |= {i, j}
    G = nx.Graph()
    G.add_edges_from(raw_edges)
    clusters = list(nx.connected_components(G))

    reps = [list(c)[0] if len(c) > cluster_threshold else list(c) for c in clusters]
    reps = [r for sub in reps for r in (r if isinstance(r, list) else [r])]

    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            a, b = reps[i], reps[j]
            if sim_matrix[a, b] > threshold:
                optimized_edges.append(go.Scattergl(
                    x=[df.iloc[a]['x'], df.iloc[b]['x']],
                    y=[df.iloc[a]['y'], df.iloc[b]['y']],
                    mode='lines',
                    line=dict(color='gray', width=0.5),
                    opacity=0.3,
                    hoverinfo='skip',
                    showlegend=False
                ))
    return raw_edges, optimized_edges, nodes_with_edges

# === Cache-heavy preprocessing ===
@lru_cache(maxsize=10)
def preprocess_data(file_tuple):
    df = load_and_merge_csvs(list(file_tuple))
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Code'])

    reducer = umap.UMAP(n_components=2, random_state=42, spread=4, min_dist=4)
    embedding = reducer.fit_transform(X)
    df['x'], df['y'] = embedding[:, 0], embedding[:, 1]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    df['Topic'] = clusterer.fit_predict(embedding)

    unique_topics = sorted(df['Topic'].unique())
    colors = (sns.color_palette("husl", len(unique_topics)).as_hex()
              if len(unique_topics) > len(topic_colors) else topic_colors[:len(unique_topics)])
    topic_color_map = {t: colors[i] for i, t in enumerate(unique_topics)}
    df['TopicColor'] = df['Topic'].map(lambda t: topic_color_map.get(t, '#999999'))

    sim_matrix = cosine_similarity(embedding)
    raw_edges, optimized_edges, connected = generate_edges(df, sim_matrix)

    return df, optimized_edges, connected, topic_color_map

# === Dash App ===
app = dash.Dash(__name__, assets_folder='assets', suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Code Similarity Visualisation"),
    html.Div([
        html.Label("Plot Title"), dcc.Input(id='plot-title', value='Code Similarity Visualisation', type='text', style={'width': '50%'}),
        html.Label("Select File(s)"), dcc.Dropdown(
            id='file-selector',
            options=[{'label': os.path.basename(f), 'value': f} for f in csv_files],
            value=[csv_files[0]],
            multi=True
        )
    ]),
    html.Button("Download as PNG", id="download-btn", n_clicks=0),
    html.Div([
        dcc.Checklist(
            id='filter-nodes',
            options=[
                {'label': 'Show only connected', 'value': 'only_connected'},
                {'label': 'Show only isolated', 'value': 'only_isolated'},
                {'label': 'Show edges', 'value': 'show_edges'}
            ],
            value=['only_connected']
        ),
        dcc.RadioItems(
            id='color-mode',
            options=[
                {'label': 'LLM', 'value': 'llm'},
                {'label': 'Prompt Type', 'value': 'prompt'},
                {'label': 'Topic', 'value': 'topic'},
                {'label': 'Class', 'value': 'class'}
            ],
            value='topic',
            inline=True
        )
    ]),
    dcc.Graph(id='scatter-plot', config={'scrollZoom': True}, style={'height': '750px'}),
    html.Div(id='output-div', style={'marginTop': 20, 'fontSize': 20})
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('file-selector', 'value'),
    Input('filter-nodes', 'value'),
    Input('color-mode', 'value'),
    Input('plot-title', 'value')
)
def update_graph(files, filters, color_mode, plot_title):
    if not files: return go.Figure()
    df, opt_edges, connected, topic_colors = preprocess_data(tuple(files))

    if 'only_connected' in filters:
        df_filtered = df.iloc[list(connected)]
    elif 'only_isolated' in filters:
        df_filtered = df[~df.index.isin(connected)]
    else:
        df_filtered = df

    # Choose color scheme
    if color_mode == 'llm':
        color_col, legend_map = 'LLMColor', {k: llm_colors[k] for k in df_filtered['LLM'].unique()}
    elif color_mode == 'prompt':
        color_col = 'PromptColor'
        pt_map = {1: "Copy Paste", 2: "Perfect", 3: "Memetic Proxy", 4: "Meta", 5: "Restraints", 6: "Translation"}
        legend_map = {pt_map.get(pt, f"Type {pt}"): num_to_color(pt) for pt in df_filtered['PromptType'].dropna().unique()}
    elif color_mode == 'class':
        color_col = 'ClassColor'
        class_map = {
            "classified_LLM": "Classified LLM", "classified_Student": "Classified Student",
            "misclassified_LLM": "Misclassified LLM", "misclassified_Student": "Misclassified Student"
        }
        legend_map = {class_map.get(c, c): class_colors[c] for c in df_filtered['Class'].unique()}
    else:
        color_col = 'TopicColor'
        legend_map = {f"Topic {t}": topic_colors[t] for t in topic_colors}
        legend_map["Noise / Unclustered"] = '#999999'

    # Legend
    legend_traces = [
        go.Scattergl(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            name=label
        )
        for label, color in legend_map.items()
    ]

    scatter = go.Scattergl(
        x=df_filtered['x'], y=df_filtered['y'],
        mode='markers',
        marker=dict(size=10, color=df_filtered[color_col], opacity=0.6),
        text=df_filtered['ID'], customdata=df_filtered['ID'],
        hoverinfo='text', showlegend=False
    )

    fig = go.Figure(data=legend_traces + (opt_edges if 'show_edges' in filters else []) + [scatter])
    fig.update_layout(
        title=plot_title, xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10), showlegend=True,
        paper_bgcolor='#f4f4f4', plot_bgcolor='#ffffff'
    )
    return fig

@app.callback(
    Output('output-div', 'children'),
    Input('scatter-plot', 'clickData')
)
def display_code(clickData):
    if clickData and 'points' in clickData:
        selected_id = clickData['points'][0]['customdata']
        for files in csv_files:
            if os.path.exists(files):
                df = load_and_merge_csvs([files])
                if selected_id in df['ID'].values:
                    code = df[df['ID'] == selected_id].iloc[0]['Code']
                    return html.Div([
                        html.P(f"Selected ID: {selected_id}", style={'fontWeight': 'bold'}),
                        html.Pre(code, style={'backgroundColor': '#f8f8f8', 'padding': '10px'})
                    ])
    return "Click a point to see its code."

app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='download_plot'
    ),
    Output('download-btn', 'n_clicks'),
    Input('download-btn', 'n_clicks')
)

if __name__ == '__main__':
    app.run(debug=True)
