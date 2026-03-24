import numpy as np
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from tangerine.visualise.data_loader import DataLoader
import matplotlib as mpl
import networkx as nx
import pandas as pd
import itertools

def make_app_layout(app, tf_list, gene_list, timepoints):
    t0 = timepoints[0]
    t1 = timepoints[1] if len(timepoints) > 1 else timepoints[0]
    tf_chosen = tf_list[0] if tf_list else None
    gene_chosen = gene_list[0] if gene_list else None

    tf_list.sort()
    gene_list.sort()

    app.layout = dbc.Container([
        # --- HEADER ---
        dbc.Card([
            dbc.CardBody([
                html.H2("🍊 Tangerine: Dynamic Gene Regulatory Network Explorer", className="text-primary"),
                html.P("Explore time-varying TF co-regulation and regulatory network topology.", className="mb-0")
            ])
        ], className="mb-4 mt-3 shadow-sm"),

        # --- TABS ---
        dbc.Tabs([
            # ==========================================================
            # TAB 1: GLOBAL TOPOLOGY & MODULES
            # ==========================================================
            dbc.Tab(label="Global Topology & Modules", tab_id="tab-1", children=[
                dbc.Row([
                    # LEFT COLUMN: Heatmap, Alluvial & Gene Chips (Width 7)
                    dbc.Col([
                        # Top Left: Pre-clustered Correlation Heatmap
                        dbc.Row([
                            dbc.Col([
                                html.H5("TF Correlation", className="mt-3"),
                                html.P("Select a timepoint. Box-select a region to track TFs.", className="text-muted small"),
                                
                                dcc.Slider(
                                    0, len(timepoints)-1, step=1, 
                                    marks={i: t for i, t in enumerate(timepoints)}, 
                                    value=0, id='heatmap-time-slider', className="mb-2"
                                ),
                                dcc.Graph(id='correlation-heatmap', style={'height': '350px'})
                            ])
                        ]),
                        
                        # Middle Left: Alluvial
                        dbc.Row([
                            dbc.Col([
                                html.H5("Module Evolution", className="mt-3"),
                                html.P("Tracks cluster membership of TFs selected above.", className="text-muted small"),
                                dcc.Graph(id='alluvial-plot', style={'height': '280px'})
                            ])
                        ]),
                        
                        # Bottom Left: Gene Chip Bank
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(
                                        dbc.Row([
                                            dbc.Col(html.Strong(id='selected-tf-count', children="0 TFs Selected")),
                                            dbc.Col([
                                                html.Span("Copy", className="text-muted small me-2"),
                                                dcc.Clipboard(
                                                    id="copy-clipboard", 
                                                    style={"display": "inline-block", "fontSize": "20px", "color": "#E67E22", "cursor": "pointer"}
                                                )
                                            ], width="auto", className="d-flex align-items-center")
                                        ], justify="between")
                                    ),
                                    dbc.CardBody(id='gene-chip-container', style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'minHeight': '60px'})
                                ], className="shadow-sm")
                            ])
                        ], className="mt-3")
                        
                    ], width=7, className="border-end pe-4"),

                    # RIGHT COLUMN: Differential Circular Graph & Scatterplots (Width 5)
                    dbc.Col([
                        html.H5("Differential Network Topology", className="mt-3"),
                        html.P("Compare two timepoints. Edges show change in Spearman correlation.", className="text-muted small"),
                        
                        dbc.InputGroup([
                            dbc.InputGroupText("Compare:"),
                            dbc.Select(options=[{'label': t, 'value': t} for t in timepoints], value=t0, id='diff-time-1'),
                            dbc.InputGroupText("vs"),
                            dbc.Select(options=[{'label': t, 'value': t} for t in timepoints], value=t1, id='diff-time-2'),
                        ], className="mb-3", size="sm"),
                        
                        html.Label("Δ Correlation Threshold (Filter Noise):", className="fw-bold small"),
                        dcc.Slider(
                            0.1, 1.0, step=0.05, value=0.75, id='delta-threshold', className="mb-2"
                        ),
                        
                        # Circular Graph
                        dcc.Graph(id='differential-circular-graph', style={'height': '500px'}),
                        
                        # Metacell Scatterplots (Side-by-side)
                        html.H6("Raw Metacell Expression (Click an edge above)", className="mt-2 text-center"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='scatter-t1', style={'height': '250px'}), width=6, className="px-1"),
                            dbc.Col(dcc.Graph(id='scatter-t2', style={'height': '250px'}), width=6, className="px-1")
                        ])
                    ], width=5, className="ps-4")
                ], className="mb-5")
            ]),

            # ==========================================================
            # TAB 2: TARGETED DYNAMICS
            # ==========================================================
            dbc.Tab(label="Targeted Dynamics", tab_id="tab-2", children=[
                # Section A: 1D Temporal Heatmap
                dbc.Row([
                    dbc.Col([
                        html.H4("TF Co-regulation Dynamics", className="mt-4"),
                        html.P("How does a specific TF's relationship with its top partners evolve?", className="text-muted"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Target TF:"),
                            dbc.Select(options=[{'label': tf, 'value': tf} for tf in tf_list], value=tf_chosen, id='ego-tf-dropdown')
                        ], className="mb-3", style={"width": "300px"}),
                        dcc.Graph(id='ego-heatmap', style={'height': '500px'})
                    ], width=12)
                ]),
                
                html.Hr(className="my-4"),

                # Section B: Split Streamgraph
                dbc.Row([
                    dbc.Col([
                        html.H4("Target Gene Regulation"),
                        html.P("Total upstream influence on a target gene over time (Ridge Regression).", className="text-muted"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Target Gene:"),
                            dbc.Select(options=[{'label': g, 'value': g} for g in gene_list], value=gene_chosen, id='gene-picker')
                        ], className="mb-3", style={"width": "300px"}),
                        dcc.Graph(id='split-streamgraph', style={'height': '700px'})
                    ], width=12)
                ], className="mb-5")
            ])
            
        ], id="tabs", active_tab="tab-1", className="mt-3")
    ], fluid=True, className="px-5")


def run_app(timepoints, base_path):
    data_loader = DataLoader(timepoints, base_path)

    # Calculate global max coefficient for consistent dot plot colors 
    global_max_coef = 0.0
    for time in timepoints:
        for u, v, data in data_loader.networks[time].edges(data=True):
            val = abs(data.get('coefficient', 0))
            if val > global_max_coef:
                global_max_coef = val
                
    if global_max_coef == 0: global_max_coef = 1.0 # Fallback safety

    t0 = timepoints[0]
    tf_list = data_loader.tf_list
    gene_list = []
    for node in data_loader.networks[t0]:
        if len(data_loader.networks[t0].in_edges(node)) > 0:
            gene_list.append(node)

    # --- Precompute Circular Layout for Consensus Graph ---
    G_base = data_loader.networks[t0]
    tf_subgraph = G_base.subgraph([n for n in G_base.nodes if n in tf_list])
    circular_pos = nx.circular_layout(tf_subgraph)

    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    make_app_layout(app, tf_list, gene_list, timepoints)

    # =========================================================================
    # CALLBACKS: TAB 1
    # =========================================================================
    
    @app.callback(
        Output('correlation-heatmap', 'figure'),
        Input('heatmap-time-slider', 'value')
    )
    def update_heatmap(time_idx):
        time = timepoints[time_idx]
        df = data_loader.tf_correlation_dfs[time]

        fig = go.Figure(go.Heatmap(
            z=df.values, x=df.columns, y=df.index,
            colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title="Spearman"),
            hovertemplate="TF X: %{x}<br>TF Y: %{y}<br>Corr: %{z:.2f}<extra></extra>"
        ))
        
        # yaxis_autorange='reversed' fixes the diagonal orientation 
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_showticklabels=False, yaxis_showticklabels=False,
            yaxis_autorange='reversed', 
            dragmode='select', plot_bgcolor='white'
        )
        return fig

    @app.callback(
        Output('alluvial-plot', 'figure'),
        Output('gene-chip-container', 'children'),
        Output('selected-tf-count', 'children'),
        Output('copy-clipboard', 'content'),
        Input('correlation-heatmap', 'selectedData'),
        State('correlation-heatmap', 'figure')
    )
    def update_alluvial_and_chips(selectedData, heatmap_fig):
        dimensions = data_loader.get_parcat_dimensions()
        colorscale_parcat = [[0, 'lightgray'], [1, '#E67E22']] 
        color_array = np.zeros(len(data_loader.tf_louvain), dtype='uint8')

        selected_tfs = []
        
        if selectedData and heatmap_fig and 'data' in heatmap_fig:
            axis_labels = heatmap_fig['data'][0]['x'] 
            
            if 'points' in selectedData and len(selectedData['points']) > 0 and 'x' in selectedData['points'][0] and isinstance(selectedData['points'][0]['x'], str):
                selected_tfs = list(set([p['x'] for p in selectedData['points']] + [p['y'] for p in selectedData['points']]))
                
            elif 'range' in selectedData:
                x_range = selectedData['range']['x']
                y_range = selectedData['range']['y']
                
                def get_selected_labels(labels, sel_range):
                    res = []
                    # Added safety min/max to handle reversed Y-axis selection ranges
                    min_val, max_val = min(sel_range), max(sel_range)
                    for i, label in enumerate(labels):
                        if (i + 0.5) >= min_val and (i - 0.5) <= max_val:
                            res.append(label)
                    return res
                    
                sel_x = get_selected_labels(axis_labels, x_range)
                sel_y = get_selected_labels(axis_labels, y_range)
                selected_tfs = list(set(sel_x + sel_y))

        if selected_tfs:
            new_indices = [data_loader.tf_louvain.index.get_loc(tf) for tf in selected_tfs if tf in data_loader.tf_louvain.index]
            color_array[new_indices] = 1

        fig = go.Figure(go.Parcats(
            dimensions=dimensions,
            line={'colorscale': colorscale_parcat, 'cmin': 0, 'cmax': 1, 'color': color_array, 'shape': 'hspline'},
            hoverinfo='none'
        ))
        fig.update_layout(margin=dict(l=20, r=40, t=20, b=20))

        if not selected_tfs:
            chips = [html.Span("Draw a box on the heatmap to select TFs.", className="text-muted")]
        else:
            chips = [dbc.Badge(tf, color="secondary", className="me-1 fs-6") for tf in selected_tfs]

        # Generate Gene Chips
        if not selected_tfs:
            chips = [html.Span("Draw a box on the heatmap to select TFs.", className="text-muted")]
            tf_string = "" # Empty clipboard
        else:
            chips = [dbc.Badge(tf, color="secondary", className="me-1 fs-6") for tf in selected_tfs]
            # Join the selected TFs with a newline character so they paste cleanly into GO tools
            tf_string = "\n".join(selected_tfs) 

        return fig, chips, f"{len(selected_tfs)} TFs Selected", tf_string


    @app.callback(
        Output('differential-circular-graph', 'figure'),
        Input('diff-time-1', 'value'),
        Input('diff-time-2', 'value'),
        Input('delta-threshold', 'value')
    )
    def update_diff_graph(t1, t2, threshold):
        if not t1 or not t2: return go.Figure()
        
        df1 = data_loader.tf_correlation_dfs[t1]
        df2 = data_loader.tf_correlation_dfs[t2]
        delta_df = df2 - df1 
        delta_df = delta_df.fillna(0)
        
        pos_edge_x, pos_edge_y = [], []
        neg_edge_x, neg_edge_y = [], []
        
        hover_x, hover_y, hover_text, hover_customdata = [], [], [], []
        active_nodes = set() # Keep track of nodes that pass the filter
        
        nodes = list(circular_pos.keys())
        
        for u, v in itertools.combinations(nodes, 2):
            if u in delta_df.index and v in delta_df.columns:
                delta = delta_df.loc[u, v]
                
                if abs(delta) >= threshold:
                    active_nodes.update([u, v]) # Mark these nodes to be drawn
                    
                    x0, y0 = circular_pos[u]
                    x1, y1 = circular_pos[v]
                    
                    if delta > 0:
                        pos_edge_x.extend([x0, x1, None])
                        pos_edge_y.extend([y0, y1, None])
                    else:
                        neg_edge_x.extend([x0, x1, None])
                        neg_edge_y.extend([y0, y1, None])
                    
                    hover_x.append((x0 + x1) / 2)
                    hover_y.append((y0 + y1) / 2)
                    hover_text.append(f"<b>{u} ↔ {v}</b><br>Δ Corr: {delta:.3f}<br><i>Click for expression</i>")
                    hover_customdata.append([u, v]) # Store the TF names for the click event

        pos_edge_trace = go.Scatter(
            x=pos_edge_x, y=pos_edge_y, mode='lines',
            line=dict(width=1.5, color="rgba(231, 76, 60, 0.7)"), hoverinfo='none'
        )

        neg_edge_trace = go.Scatter(
            x=neg_edge_x, y=neg_edge_y, mode='lines',
            line=dict(width=1.5, color="rgba(52, 152, 219, 0.7)"), hoverinfo='none'
        )
        
        edge_hover_trace = go.Scatter(
            x=hover_x, y=hover_y, mode='markers',
            marker=dict(size=12, color='rgba(0,0,0,0)'),
            text=hover_text,
            customdata=hover_customdata, # Pass data to click event
            hovertemplate="%{text}<extra></extra>"
        )

        # Only draw nodes that are in active_nodes!
        node_x = [circular_pos[n][0] for n in active_nodes]
        node_y = [circular_pos[n][1] for n in active_nodes]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            marker=dict(size=6, color='#2C3E50', line=dict(width=1, color='white')),
            text=list(active_nodes), 
            textposition="top center",
            textfont=dict(size=9),
            hoverinfo='none'
        )

        fig = go.Figure(data=[pos_edge_trace, neg_edge_trace, edge_hover_trace, node_trace])
        fig.update_layout(
            showlegend=False, hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        return fig
    
    @app.callback(
        Output('scatter-t1', 'figure'),
        Output('scatter-t2', 'figure'),
        Input('differential-circular-graph', 'clickData'),
        State('diff-time-1', 'value'),
        State('diff-time-2', 'value')
    )
    def update_scatterplots(clickData, t1, t2):
        # Empty state before user clicks an edge
        empty_fig = go.Figure().update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text="Click an edge to view<br>metacell expression.", xref="paper", yref="paper", showarrow=False, font=dict(color="gray"))],
            plot_bgcolor='white', margin=dict(l=0, r=0, t=30, b=0)
        )
        if not clickData or not t1 or not t2:
            return empty_fig, empty_fig

        # Extract which TFs were clicked from the customdata we set earlier
        tf_u, tf_v = clickData['points'][0]['customdata']

        def create_scatter(time, tf_x, tf_y):
            # Fetch exactly two columns from disk instantly
            df_exp = data_loader.get_metacell_expression(time, [tf_x, tf_y])

            # Fetch the exact Spearman correlation
            try:
                corr_val = data_loader.tf_correlation_dfs[time].loc[tf_x, tf_y]
            except KeyError:
                corr_val = 0.0 # Safety fallback
            
            # Safety check: if file is missing or TFs aren't in columns, return empty arrays
            if df_exp.empty or tf_x not in df_exp.columns or tf_y not in df_exp.columns:
                exp_x, exp_y = [], []
            else:
                exp_x = df_exp[tf_x].values
                exp_y = df_exp[tf_y].values

            fig = go.Figure(go.Scatter(
                x=exp_x, y=exp_y, mode='markers',
                marker=dict(size=5, color='#E67E22', opacity=1, line=dict(width=0.5, color='white'))
            ))
            fig.update_layout(
                title=dict(text=f"<b>{time}</b> (Spearman ρ = {corr_val:.2f})", font=dict(size=12), x=0.5, xanchor='center'),
                xaxis_title=dict(text=tf_x, font=dict(size=10)),
                yaxis_title=dict(text=tf_y, font=dict(size=10)),
                margin=dict(l=30, r=10, t=30, b=30),
                plot_bgcolor='rgba(240,240,240,0.5)' # Slight grey background to pop the dots
            )
            return fig

        fig_t1 = create_scatter(t1, tf_u, tf_v)
        fig_t2 = create_scatter(t2, tf_u, tf_v)

        return fig_t1, fig_t2


    # =========================================================================
    # CALLBACKS: TAB 2
    # =========================================================================

    @app.callback(
        Output('ego-heatmap', 'figure'),
        Input('ego-tf-dropdown', 'value')
    )
    def update_1d_heatmap(tf_name):
        if not tf_name: return go.Figure()
        
        ego_data = {t: data_loader.tf_correlation_dfs[t].loc[tf_name] for t in timepoints}
        ego_df = pd.DataFrame(ego_data)
        
        if tf_name in ego_df.index: ego_df = ego_df.drop(index=tf_name)
        top_tfs = ego_df.abs().max(axis=1).nlargest(15).index
        ego_top_df = ego_df.loc[top_tfs]

        fig = go.Figure(go.Heatmap(
            z=ego_top_df.values, x=ego_top_df.columns, y=ego_top_df.index,
            colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
            text=np.round(ego_top_df.values, 2), texttemplate="%{text}", textfont={"size":10},
            colorbar=dict(title="Spearman")
        ))
        
        # Apply the same Y-axis reversal here so the labels align logically
        fig.update_layout(
            margin=dict(l=100, r=20, t=20, b=40), 
            yaxis_autorange='reversed',
            plot_bgcolor='white'
        )
        return fig

    @app.callback(
        Output('split-streamgraph', 'figure'), 
        Input('gene-picker', 'value')
    )
    def update_temporal_dot_plot(gene_name):
        if not gene_name: return go.Figure()

        df_coef = data_loader.get_in_edges_dataframe(gene_name, 'coefficient')
        if df_coef.empty: return go.Figure()
        
        if 'avg' in df_coef.columns: df_coef = df_coef.drop(columns=['avg'])
        
        noise_threshold = 1e-2
        df_coef = df_coef.loc[(df_coef.abs() > noise_threshold).any(axis=1)]

        x_vals, y_vals, coef_vals, size_vals = [], [], [], []

        for tf in df_coef.index:
            for time in timepoints:
                val = df_coef.loc[tf, time]
                
                if pd.notna(val) and abs(val) > noise_threshold:
                    x_vals.append(time)
                    y_vals.append(tf)
                    coef_vals.append(val)
                    
                    normalized_val = abs(val) / global_max_coef
                    exaggerated_size = (normalized_val ** 0.75) * 55
                    size_vals.append(max(exaggerated_size, 8)) 

        if x_vals: 
            sorted_data = sorted(zip(x_vals, y_vals, coef_vals, size_vals), key=lambda item: item[3], reverse=True)
            x_vals, y_vals, coef_vals, size_vals = map(list, zip(*sorted_data))
        
        fig = go.Figure(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(
                size=size_vals,
                color=coef_vals,
                colorscale='RdBu_r', 
                cmin=-global_max_coef, 
                cmax=global_max_coef,  
                showscale=True,
                opacity=1.0,           
                colorbar=dict(title="Ridge<br>Coefficient"),
                line=dict(width=1.5, color='white') 
            ),
            text=coef_vals,
            hovertemplate="<b>TF:</b> %{y}<br><b>Time:</b> %{x}<br><b>Coefficient:</b> %{text:.3f}<extra></extra>"
        ))

        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=40),
            xaxis=dict(title="Timepoints", showgrid=True, gridcolor='rgba(0,0,0,0.05)', tickmode='array', tickvals=timepoints),
            yaxis=dict(title="Upstream TFs", showgrid=True, gridcolor='rgba(0,0,0,0.05)', autorange='reversed'), 
            plot_bgcolor='white',
            hovermode='closest'
        )
        
        return fig

    app.run(debug=True)