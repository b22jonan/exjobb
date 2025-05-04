import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pyperclip
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
import networkx as nx
import hdbscan
import seaborn as sns 


# Define color mapping for LDA topics
topic_colors = ['#DC143C', '#40E0D0', '#E97451', '#8B00FF', '#DAA520', '#009688', '#FF4500', '#FFD700', '#FF69B4', '#00FF00']
# Define color mapping for PromptType
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
# Define LLM Color Mapping
llm_colors = {
    "ChatGPT4o": "#FF4500",      # Orange Red
    "ChatGPT35": "#1E90FF",      # Dodger Blue
    "DeepSeek": "#32CD32",   # Lime Green
    "Qwen": "#DFB700"        # Yellow
}

class_colors = {
    "classified_LLM": "#666666",  # Grey
    "classified_Student":  "#FF4500",  # Orange Red
    "misclassified_Student": "#ff00ff",  # Magenta
    "misclassified_LLM": "#DFB700",  # Yellow
}

# Directory containing CSV files
csv_folder = 'ML_models/code_similarity/files_in_use/'
csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]

# Extract LLM Name from Filename
def extract_llm_from_filename(filename):
    return filename.split('_')[-1].replace('.csv', '')

# Extract class from filename
# updated_classified_LLM_AdaBoost_ChatGPT4o.csv -> classified_LLM
def extract_class_from_filename(filename):    
    parts = filename.split('_')
    if len(parts) > 2:
        return '_'.join(parts[1:-2])  # Join parts excluding the first and last two
    return filename

# Load and merge all CSV files
def load_and_merge_csvs(csv_files):
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)  # Track source file
        
        # Drop "prompt" column to ensure uniform comparisons
        if 'prompt' in df.columns:
            df = df.drop(columns=['prompt'])
        
        df['LLM'] = extract_llm_from_filename(os.path.basename(file))  # Extract LLM name
        df['LLMColor'] = df['LLM'].map(llm_colors)  # Assign LLM color
        
        df['Class'] = extract_class_from_filename(os.path.basename(file))  # Extract class name
        df['ClassColor'] = df['Class'].map(class_colors)  # Assign class color
        
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['ID'])

# Load all data
df = load_and_merge_csvs(csv_files)

# Assign PromptType colors
df['PromptColor'] = df['PromptType'].map(num_to_color)

# Vectorize ONLY the "Code" column
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Code'])  # This ensures only "Code" is used

# **Reduce TF-IDF Vectors to 2D using UMAP**
umap_reducer = umap.UMAP(n_components=2, random_state=42, spread=4, min_dist=4)
X_embedded = umap_reducer.fit_transform(X)  # Use TF-IDF directly instead of LDA

# **Apply HDBSCAN to dynamically find the number of clusters**
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
df['Topic'] = clusterer.fit_predict(X_embedded)  # Assign dynamic topics

# Get sorted unique topic labels (to handle gaps in numbering)
unique_topics = sorted(df['Topic'].unique())  # This ensures topic indices are ordered

# Ensure enough colors are available (dynamically generate if needed)
if len(unique_topics) > len(topic_colors):  
    dynamic_topic_colors = sns.color_palette("husl", len(unique_topics)).as_hex()  # Generate colors
else:
    dynamic_topic_colors = topic_colors[:len(unique_topics)]  # Use predefined colors

# Create a mapping from topic labels (which can be non-sequential) to color indices
topic_to_color_map = {topic: dynamic_topic_colors[i] for i, topic in enumerate(unique_topics)}

# Assign topic colors safely
df['TopicColor'] = df['Topic'].map(lambda t: topic_to_color_map.get(t, '#999999'))  # Default to gray for noise


## Offset overlapping points in a consistent direction
#def offset_duplicates(coords, min_distance=0.5):
#    seen = {}
#    adjusted_coords = coords.copy()
#    offset_step = min_distance
#    for i, point in enumerate(coords):
#        key = tuple(np.round(point, decimals=3))
#        if key in seen:
#            count = seen[key]
#            adjusted_coords[i] += np.array([offset_step * count, offset_step * count])
#            seen[key] += 1
#        else:
#            seen[key] = 1
#    return adjusted_coords
#
#X_embedded = offset_duplicates(X_embedded)

df['x'] = X_embedded[:, 0]
df['y'] = X_embedded[:, 1]

# Set similarity threshold for edges (Reduce for performance)
threshold = 0.5
nodes_with_edges = set()
df = df.reset_index(drop=True)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
labels = clusterer.fit_predict(X_embedded)  # Use UMAP output instead of LDA

df['Topic'] = labels  # Assign cluster labels to dataframe

# Compute similarity using the UMAP-reduced features instead
similarity_matrix = cosine_similarity(X_embedded)


# Function to generate both full & optimized edges
def generate_edges(df, similarity_matrix, threshold=0.5, cluster_threshold=5):
    """
    Generates:
    - raw_edges (full edges, used for filtering)
    - optimized_edges (cluster-based edges, planned on using for rendering, doesnt work)

    Args:
        df: DataFrame of nodes.
        similarity_matrix: Precomputed similarity matrix.
        threshold: Similarity threshold for edges.
        cluster_threshold: Min cluster size before collapsing edges.

    Returns:
        raw_edges: Full edge set (for filtering)
        optimized_edges: Simplified edges (for rendering)
        nodes_with_edges: Set of nodes that have edges
    """
    raw_edges = []  # Full edges (for filtering)
    optimized_edges = []  # Reduced edges (for rendering)
    nodes_with_edges = set()

    # **Full Edge List (for filtering)**
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if similarity_matrix[i, j] > threshold:
                raw_edges.append((i, j))  # Store raw edges
                nodes_with_edges.add(i)
                nodes_with_edges.add(j)

    # **Cluster-Based Edge Simplification**
    G = nx.Graph()
    for i in range(len(df)):
        G.add_node(i)

    for i, j in raw_edges:
        G.add_edge(i, j, weight=similarity_matrix[i, j])

    # Find clusters (connected components)
    clusters = list(nx.connected_components(G))
    
    # Choose representative nodes for clusters
    cluster_representatives = []
    for cluster in clusters:
        cluster = list(cluster)
        if len(cluster) > cluster_threshold:
            representative = cluster[0]  # Choose first node as representative
            cluster_representatives.append(representative)
        else:
            cluster_representatives.extend(cluster)

    # Build a new graph using only cluster representatives
    simplified_G = nx.Graph()
    for rep in cluster_representatives:
        simplified_G.add_node(rep)

    for i in range(len(cluster_representatives)):
        for j in range(i + 1, len(cluster_representatives)):
            node_i = cluster_representatives[i]
            node_j = cluster_representatives[j]
            if similarity_matrix[node_i, node_j] > threshold:
                simplified_G.add_edge(node_i, node_j, weight=similarity_matrix[node_i, node_j])

    # Extract edges from the simplified graph (for rendering)
    for edge in simplified_G.edges():
        i, j = edge
        optimized_edges.append(go.Scattergl(
            x=[df.iloc[i]['x'], df.iloc[j]['x']],
            y=[df.iloc[i]['y'], df.iloc[j]['y']],
            mode='lines',
            line=dict(color='gray', width=0.5),
            opacity=0.3,
            hoverinfo='skip',
            showlegend=False
        ))

    return raw_edges, optimized_edges, nodes_with_edges

# Generate edges
raw_edges, optimized_edges, nodes_with_edges = generate_edges(df, similarity_matrix, threshold=0.5, cluster_threshold=5)

# Assign colors to the dataframe
df['PromptColor'] = df['PromptType'].map(num_to_color)  # Assign PromptType colors

# Dash App
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Dash UI
app.layout = html.Div([
    html.Div(
        style={'backgroundColor': '#f4f4f4', 'padding': '10px', 'textAlign': 'center'},
        children=[
            html.H1("Code Similarity Visualisation", style={'margin': '0px'}),
            html.P("Merging all datasets for visualization."),
            dcc.Checklist(
                id='filter-nodes',
                options=[
                    {'label': 'Show only nodes with similarity edges', 'value': 'only_connected'},
                    {'label': 'Show only isolated nodes', 'value': 'only_isolated'},
                    {'label': 'Show edges', 'value': 'show_edges'}
                ],
                value=['only_connected']
            ),
            dcc.RadioItems(
                id='color-mode',
                options=[
                    {'label': 'Color by LLM', 'value': 'llm'},
                    {'label': 'Color by Prompt Type', 'value': 'prompt'},
                    {'label': 'Color by Topic', 'value': 'topic'},
                    {'label': 'Color by Class', 'value': 'class'}
                ],
                value='topic',  # Default: Topic-based coloring
                inline=True
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
    # Dynamic Legend
    html.Div(id='legend-div', style={'textAlign': 'center', 'marginTop': '20px'}),
    html.Div(id='output-div', style={'marginTop': 20, 'fontSize': 20, 'textAlign': 'center'})
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Output('legend-div', 'children'),  # Keeps legend
    [Input('filter-nodes', 'value'),
     Input('color-mode', 'value')]  # New input for color mode
)


def update_graph(filter_value, color_mode):
    show_edges = 'show_edges' in filter_value

    # Select color mode
    if color_mode == 'llm':
        color_column = 'LLMColor'
    elif color_mode == 'prompt':
        color_column = 'PromptColor'
    elif color_mode == 'topic':
        color_column = 'TopicColor'
    elif color_mode == 'class':
        color_column = 'ClassColor'
    else:
        color_column = 'whatever'  # Fallback

    # **Fix for Node Filtering**
    if 'only_connected' in filter_value:
        filtered_df = df.iloc[list(nodes_with_edges)]
    elif 'only_isolated' in filter_value:
        filtered_df = df[~df.index.isin(nodes_with_edges)]
    else:
        filtered_df = df

    figure_data = optimized_edges if show_edges else []
    figure_data.append(
        go.Scattergl(
            x=filtered_df['x'],
            y=filtered_df['y'],
            mode='markers',
            marker=dict(size=10, color=filtered_df[color_column], opacity=0.5),
            text=filtered_df['ID'],
            customdata=filtered_df['ID'],
            hoverinfo='text'
        )
    )

    # **Create the figure**
    figure = go.Figure(
        data=figure_data,
        layout=go.Layout(
            title="AdaBoost - DeepSeek Missclassified Code Similarity Visualisation",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='#f4f4f4',
            plot_bgcolor='#ffffff'
        )
    )

    # **Legend Logic**
    if color_mode == 'llm':
        legend = html.Div([
            html.P("Legend (LLM Models):", style={'fontWeight': 'bold'}),
            html.Div([
                html.Span("GPT-4o - Orange Red", style={'color': '#FF4500'}),
                html.Span(" | ", style={'color': '#000000'}),
                html.Span("GPT-3.5 - Dodger Blue", style={'color': '#1E90FF'}),
                html.Span(" | ", style={'color': '#000000'}),
                html.Span("DeepSeek - Lime Green", style={'color': '#32CD32'}),
                html.Span(" | ", style={'color': '#000000'}),
                html.Span("Qwen - Yellow", style={'color': '#DFB700'})
            ])
        ])
    elif color_mode == 'prompt':
        legend = html.Div([
            html.P("Legend (Prompt Types):", style={'fontWeight': 'bold'}),
            html.Div([
                html.Span("Copy Paste - Crimson", style={'color': '#DC143C'}),
                html.Span(" | ", style={'color': '#000000'}),
                html.Span("Perfect - Turquoise", style={'color': '#40E0D0'}),
                html.Span(" | ", style={'color': '#000000'}),
                html.Span("Memetic Proxy - Burnt Sienna", style={'color': '#E97451'}),
                html.Span(" | ", style={'color': '#000000'}),
                html.Span("Meta - Electric Violet", style={'color': '#8B00FF'}),
                html.Span(" | ", style={'color': '#000000'}),
                html.Span("Restraints - Goldenrod", style={'color': '#DAA520'}),
                html.Span(" | ", style={'color': '#000000'}),
                html.Span("Translation - Teal", style={'color': '#009688'})
            ])
        ])
    elif color_mode == 'class':
        legend = html.Div([
            html.P("Legend (Classes):", style={'fontWeight': 'bold'}),
            html.Div([
                html.Span("Classified LLM - grey", style={'color': '#666666'}),
                html.Span(" | ", style={'color': '#000000'}),
                html.Span("Classified Student - Orange Red", style={'color': '#FF4500'}),
                html.Span(" | ", style={'color': '#000000'}),
                html.Span("Misclassified Student - Magenta", style={'color': '#ff00ff'}),
                html.Span(" | ", style={'color': '#000000'}),
                html.Span("Misclassified LLM - Yellow", style={'color': '#DFB700'})
            ])
        ])
    else:
        unique_topics_sorted = sorted(df['Topic'].unique())
        legend_items = [
            html.Span(f"Topic {topic} ", style={'color': dynamic_topic_colors[idx]})
            for idx, topic in enumerate(unique_topics_sorted) if topic >= 0
        ]
        legend_items.append(html.Li("Noise / Unclustered", style={'color': '#999999'}))
        
        legend = html.Div([
            html.P("Legend (Topics):", style={'fontWeight': 'bold'}),
            html.Ul(legend_items)
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
    app.run(debug=True)
