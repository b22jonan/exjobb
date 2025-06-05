import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import ClientsideFunction
import pyperclip
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
import networkx as nx
import hdbscan
import seaborn as sns 

legend_traces = []  # predefined variable for legend traces

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

prompt_shape_map = {
    1: 'x',                  # Copy Paste
    2: 'triangle-up',        # Perfect
    3: 'square',             # Memetic Proxy
    4: 'circle',             # Meta
    5: 'diamond',            # Restraints
    6: 'cross'               # Translation
}


# Define LLM Color Mapping
llm_colors = {
    "ChatGPT4o": "#FF4500",     # Orange Red
    "ChatGPT35": "#1E90FF",     # Dodger Blue
    "DeepSeek": "#32CD32",      # Lime Green
    "Qwen": "#DFB700"           # Yellow
}

llm_shapes = {
    "ChatGPT4o": "circle",
    "ChatGPT35": "triangle-up",
    "DeepSeek": "square",
    "Qwen": "diamond",
}

class_colors = {
    "classified_LLM": "#666666",            # Dark Gray
    "misclassified_LLM": "#000000",         # Black
    "classified_Student": "#FF4500",        # Orange Red 
    "misclassified_Student": "#8B008B",     # Dark Magenta
}


class_shapes = {
    "classified_LLM": "triangle-up",
    "misclassified_LLM": "circle",
    "classified_Student": "x",
    "misclassified_Student": "square",
}

# Directory containing CSV files
csv_folder = 'ML_models/code_similarity/files_in_use/'
csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]

# Extract LLM Name from Filename
def extract_llm_from_filename(filename):
    return filename.split('_')[-1].replace('.csv', '')

# Extract class from filename
def extract_class_from_filename(filename):    
    parts = filename.split('_')
    if len(parts) > 2:
        return '_'.join(parts[1:-2])  
    return filename

# Load and merge all CSV files
def load_and_merge_csvs(csv_files):
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)  
        
        # Drop "prompt" column to ensure uniform comparisons
        if 'prompt' in df.columns:
            df = df.drop(columns=['prompt'])
        
        df['LLM'] = extract_llm_from_filename(os.path.basename(file))  
        df['LLMColor'] = df['LLM'].map(llm_colors)  
        
        df['Class'] = extract_class_from_filename(os.path.basename(file)) 
        df['ClassColor'] = df['Class'].map(class_colors)  
        
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['ID'])

# Load all data
df = load_and_merge_csvs(csv_files)

# Assign PromptType colors
df['PromptColor'] = df['PromptType'].map(num_to_color)

# Vectorize ONLY the "Code" column
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Code']) 

# **Reduce TF-IDF Vectors to 2D using UMAP**
umap_reducer = umap.UMAP(n_components=2, random_state=42, spread=4, min_dist=4)
X_embedded = umap_reducer.fit_transform(X) 

# **Apply HDBSCAN to dynamically find the number of clusters**
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
df['Topic'] = clusterer.fit_predict(X_embedded)  

# Get sorted unique topic labels (to handle gaps in numbering)
unique_topics = sorted(df['Topic'].unique()) 

# Ensure enough colors are available (dynamically generate if needed)
if len(unique_topics) > len(topic_colors):  
    dynamic_topic_colors = sns.color_palette("husl", len(unique_topics)).as_hex()
else:
    dynamic_topic_colors = topic_colors[:len(unique_topics)]  

# Create a mapping from topic labels (which can be non-sequential) to color indices
topic_to_color_map = {topic: dynamic_topic_colors[i] for i, topic in enumerate(unique_topics)}

# Assign topic colors safely
df['TopicColor'] = df['Topic'].map(lambda t: topic_to_color_map.get(t, '#999999')) 

df['x'] = X_embedded[:, 0]
df['y'] = X_embedded[:, 1]

# Set similarity threshold for edges (Reduce for performance)
threshold = 0.5
nodes_with_edges = set()
df = df.reset_index(drop=True)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
labels = clusterer.fit_predict(X_embedded)

df['Topic'] = labels 

# Compute similarity using the UMAP-reduced features instead
similarity_matrix = cosine_similarity(X_embedded)


# Function to generate both full & optimized edges
def generate_edges(df, similarity_matrix, threshold=0.5, cluster_threshold=5):
   
    raw_edges = []  
    optimized_edges = []  
    nodes_with_edges = set()

    # **Full Edge List (for filtering)**
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if similarity_matrix[i, j] > threshold:
                raw_edges.append((i, j))  
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
            representative = cluster[0]
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
df['PromptColor'] = df['PromptType'].map(num_to_color)

# Dash App
app = dash.Dash(__name__, assets_folder='assets', suppress_callback_exceptions=True)

# Dash UI
app.layout = html.Div([
    html.Div(
        style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '20px'},
        children=[
            html.Div([
                html.H1("Code Similarity Visualisation", style={'fontSize': '32px', 'margin': '0'}),
                html.P("Merging all datasets for visualization.", style={'fontSize': '16px'}),
            ]),
        ]
    ),
    html.Div([
        dcc.Checklist(
            id='filter-nodes',
            options=[
                {'label': 'Show only nodes with similarity edges', 'value': 'only_connected'},
                {'label': 'Show only isolated nodes', 'value': 'only_isolated'},
                {'label': 'Show edges', 'value': 'show_edges'}
            ],
            value=['only_connected'],
            style={'marginBottom': '10px'}
        ),
        dcc.RadioItems(
            id='color-mode',
            options=[
                {'label': 'Color by LLM', 'value': 'llm'},
                {'label': 'Color by Prompt Type', 'value': 'prompt'},
                {'label': 'Color by Topic', 'value': 'topic'},
                {'label': 'Color by Class', 'value': 'class'}
            ],
            value='topic',
            inline=True,
            style={'marginBottom': '10px'}
        ),
        html.Button("Download as PNG", id="download-btn", n_clicks=0),
        dcc.Graph(
            id='scatter-plot',
            style={'height': '750px', 'width': '100%'},
            config={'scrollZoom': True, 'displayModeBar': True}
        )
    ], style={'padding': '20px'}),
    
    html.Div(id='output-div', style={'marginTop': 20, 'fontSize': 20, 'textAlign': 'center'})
])


@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('filter-nodes', 'value'),
     Input('color-mode', 'value')] 
)


def update_graph(filter_value, color_mode):
    show_edges = 'show_edges' in filter_value

    # 1. Filter nodes
    if 'only_connected' in filter_value:
        filtered_df = df.iloc[list(nodes_with_edges)]
    elif 'only_isolated' in filter_value:
        filtered_df = df[~df.index.isin(nodes_with_edges)]
    else:
        filtered_df = df

    # 2. Determine color column
    if color_mode == 'llm':
        color_column = 'LLMColor'
        legend_items = {
            llm: llm_colors[llm]
            for llm in filtered_df['LLM'].unique()
        }

    elif color_mode == 'prompt':
        color_column = 'PromptColor'
        prompt_type_names = {
            1: "Copy Paste",
            2: "Perfect",
            3: "Memetic Proxy",
            4: "Meta",
            5: "Restraints",
            6: "Translation"
        }
        legend_items = {
            prompt_type_names.get(pt, f"Type {pt}"): num_to_color(pt)
            for pt in sorted(filtered_df['PromptType'].dropna().unique())
        }

    elif color_mode == 'class':
        color_column = 'ClassColor'
        class_name_map = {
            "classified_LLM": "Classified LLM",
            "classified_Student": "Classified Student",
            "misclassified_Student": "Misclassified Student",
            "misclassified_LLM": "Misclassified LLM"
        }
        legend_items = {
            class_name_map.get(cls, cls): class_colors[cls]
            for cls in filtered_df['Class'].unique()
        }

    elif color_mode == 'topic':
        color_column = 'TopicColor'
        legend_items = {
            f"Topic {topic}": topic_to_color_map.get(topic, '#999999')
            for topic in sorted(filtered_df['Topic'].unique()) if topic >= 0
        }
        legend_items["Noise / Unclustered"] = "#999999"
    else:
        color_column = 'ClassColor'
        legend_items = {}

    # 3. Build legend traces
    legend_traces = [
        go.Scattergl(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=label,
            showlegend=True
        )
        for label, color in legend_items.items()
    ]

    # 4. Build plot data
    figure_data = legend_traces + (optimized_edges if show_edges else [])
    if color_mode == 'llm':
        shape_column = 'LLM'
        shape_map = llm_shapes
    elif color_mode == 'prompt':
        shape_column = 'PromptType'
        shape_map = prompt_shape_map
    elif color_mode == 'class':
        shape_column = 'Class'
        shape_map = class_shapes
    else:
        shape_column = None
        shape_map = {}

    # Add scatter traces by shape
    if shape_column:
        for shape_value, shape_symbol in shape_map.items():
            group = filtered_df[filtered_df[shape_column] == shape_value]
            if not group.empty:
                figure_data.append(
                    go.Scattergl(
                        x=group['x'],
                        y=group['y'],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color=group[color_column],
                            symbol=shape_symbol,
                            opacity=0.6
                        ),
                        text=group['ID'],
                        customdata=group['ID'],
                        hoverinfo='text',
                        showlegend=False
                    )
                )
    else:
        # Default single shape if no shape mapping
        figure_data.append(
            go.Scattergl(
                x=filtered_df['x'],
                y=filtered_df['y'],
                mode='markers',
                marker=dict(size=10, color=filtered_df[color_column], opacity=0.5),
                text=filtered_df['ID'],
                customdata=filtered_df['ID'],
                hoverinfo='text',
                showlegend=False
            )
        )


    # 5. Create the figure
    figure = go.Figure(
        data=figure_data,
        layout=go.Layout(
            title={
                "text" : "AdaBoost - DeepSeek Classification Visualisation",
                "font": {"size": 20}
                },
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='#f4f4f4',
            plot_bgcolor='#ffffff',
            showlegend=True,
            legend=dict(
                x=1,
                y=1,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=25)
            ),
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
