import numpy as np
from dash import Dash, html, dcc, callback, Output, Input, State, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
                        
                    ], width=6, className="border-end pe-4"),

                    # RIGHT COLUMN: Dynamic Module Tracking (Width 5)
                    dbc.Col([
                        html.H5("Dynamic Module Tracking", className="mt-3"),
                        html.P("Draw a box on the global heatmap to import genes, then select timepoints to compare.", className="text-muted small"),
                        
                        # The Editable Dropdown (Single Source of Truth)
                        dcc.Dropdown(
                            id='gene-tracker-dropdown',
                            options=[{'label': g, 'value': g} for g in tf_list], 
                            value=[], 
                            multi=True,
                            placeholder="Select genes or import via heatmap...",
                            className="mb-2"
                        ),

                        # Timepoint Selectors for the Heatmaps
                        dbc.InputGroup([
                            dbc.InputGroupText("Compare:"),
                            dbc.Select(options=[{'label': t, 'value': t} for t in timepoints], value=timepoints[0], id='tracker-time-1'),
                            dbc.InputGroupText("vs"),
                            dbc.Select(options=[{'label': t, 'value': t} for t in timepoints], value=timepoints[-1], id='tracker-time-2'),
                        ], className="mb-3", size="sm"),
                        
                        # Small Multiples Plot 
                        dcc.Graph(id='small-multiples-heatmaps', style={'height': '400px', 'width': '100%'})
                    ], width=6, className="ps-4")
                ], className="mb-5")
            ]),

            # ==========================================================
            # TAB 2: TARGETED DYNAMICS
            # ==========================================================
            # ==========================================================
            # TAB 2: TARGETED DYNAMICS
            # ==========================================================
            dbc.Tab(label="Targeted Dynamics", tab_id="tab-2", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Targeted Search & Dynamics", className="mt-4"),
                        html.P("Expand a panel below to explore specific gene or TF regulatory relationships.", className="text-muted mb-4"),
                        
                        # The Accordion automatically manages expanding/collapsing
                        dbc.Accordion([
                            
                            # PANEL 1: Upstream (Target Gene)
                            dbc.AccordionItem([
                                html.P("Total upstream influence on a target gene over time (Ridge Regression).", className="text-muted small"),
                                dbc.InputGroup([
                                    dbc.InputGroupText("Target Gene:"),
                                    dbc.Select(options=[{'label': g, 'value': g} for g in gene_list], value=gene_chosen, id='gene-picker')
                                ], className="mb-3", style={"width": "300px"}),
                                
                                html.Div(dcc.Graph(id='split-streamgraph'), style={'overflowY': 'auto', 'maxHeight': '600px', 'border': '1px solid #f0f0f0'})
                            ], title="1. Upstream Regulators (Who drives this gene?)"),

                            # PANEL 2: Downstream (TF Targets)
                            dbc.AccordionItem([
                                html.P("Explore all genes regulated by a specific TF, colored by Spearman correlation.", className="text-muted small"),
                                dbc.InputGroup([
                                    dbc.InputGroupText("Regulator TF:"),
                                    dbc.Select(options=[{'label': tf, 'value': tf} for tf in tf_list], value=tf_chosen, id='downstream-tf-picker')
                                ], className="mb-3", style={"width": "300px"}),
                                
                                html.Div(dcc.Graph(id='downstream-heatmap'), style={'overflowY': 'auto', 'maxHeight': '600px', 'border': '1px solid #f0f0f0'})
                            ], title="2. Downstream Targets (What does this TF drive?)"),

                            # PANEL 3: Co-regulation (Ego Network)
                            dbc.AccordionItem([
                                html.P("How does a specific TF's relationship with all other TFs evolve over time?", className="text-muted small"),
                                dbc.InputGroup([
                                    dbc.InputGroupText("Target TF:"),
                                    dbc.Select(options=[{'label': tf, 'value': tf} for tf in tf_list], value=tf_chosen, id='ego-tf-dropdown')
                                ], className="mb-3", style={"width": "300px"}),
                                
                                html.Div(dcc.Graph(id='ego-heatmap'), style={'overflowY': 'auto', 'maxHeight': '600px', 'border': '1px solid #f0f0f0'})
                            ], title="3. TF Co-regulation Dynamics")

                        ], start_collapsed=False, always_open=False, active_item="item-0") # item-0 opens the first panel by default

                    ], width=12)
                ], className="mb-5")
            ]),

            # ==========================================================
            # TAB 3: DIFFERENTIAL TOPOLOGY (Moved from Tab 1)
            # ==========================================================
            # ==========================================================
            # TAB 3: DIFFERENTIAL TOPOLOGY
            # ==========================================================
            dbc.Tab(label="Differential Topology", tab_id="tab-3", children=[
                dbc.Row([
                    # LEFT SECTION: Controls + Circular Graph (Width 8)
                    dbc.Col([
                        html.H4("Differential Network Topology", className="mt-4"),
                        html.P("Compare two timepoints. Edges show change in Spearman correlation.", className="text-muted"),
                        
                        dbc.Row([
                            # Controls Sidebar Panel
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.Label("Compare Timepoints:", className="fw-bold small"),
                                        dbc.InputGroup([
                                            dbc.Select(options=[{'label': t, 'value': t} for t in timepoints], value=t0, id='diff-time-1'),
                                            dbc.InputGroupText("vs"),
                                            dbc.Select(options=[{'label': t, 'value': t} for t in timepoints], value=t1, id='diff-time-2'),
                                        ], className="mb-4", size="sm"),
                                        
                                        html.Label("Δ Correlation Threshold:", className="fw-bold small mb-1"),
                                        html.P("Filter noise from the network.", className="text-muted small mb-3"),
                                        dcc.Slider(
                                            0.1, 1.0, step=0.05, value=0.75, id='delta-threshold', 
                                            marks={0.2: '0.2', 0.5: '0.5', 0.8: '0.8'} # Simplified marks for a cleaner look
                                        ),
                                    ], className="p-3")
                                ], className="shadow-sm border-0 bg-light mt-4")
                            ], width=4), 
                            
                            # Circular Layout Graph 
                            dbc.Col([
                                dcc.Graph(id='differential-circular-graph', style={'height': '600px'})
                            ], width=8)
                        ])
                    ], width=8, className="border-end pe-4"),

                    # RIGHT SECTION: Stacked Scatterplots (Width 4)
                    dbc.Col([
                        html.H5("Raw Metacell Expression", className="mt-4"),
                        html.P("Click an edge in the network to view.", className="text-muted small"),
                        
                        # Stacked vertically for the sidebar layout
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='scatter-t1', style={'height': '320px'}), width=12, className="mb-3"),
                            dbc.Col(dcc.Graph(id='scatter-t2', style={'height': '320px'}), width=12)
                        ])
                    ], width=4, className="ps-4")
                ], className="mb-5 mt-2")
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
    # CALLBACKS: TAB 1 (Global Topology & Small Multiples)
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
            colorscale='RdBu_r', zmid=0, zmin=-0.75, zmax=0.75,
            colorbar=dict(title="Spearman"),
            hovertemplate="TF X: %{x}<br>TF Y: %{y}<br>Corr: %{z:.2f}<extra></extra>"
        ))
        
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
            tf_string = "" 
        else:
            chips = [dbc.Badge(tf, color="secondary", className="me-1 fs-6") for tf in selected_tfs]
            tf_string = "\n".join(selected_tfs) 

        return fig, chips, f"{len(selected_tfs)} TFs Selected", tf_string

    # Importer Callback (Heatmap -> Dropdown) 
    @app.callback(
        Output('gene-tracker-dropdown', 'value'),
        Input('correlation-heatmap', 'selectedData'),
        State('correlation-heatmap', 'figure') # Added State to read the axis labels
    )
    def import_genes_from_heatmap(selectedData, heatmap_fig):
        if not selectedData or not heatmap_fig or 'data' not in heatmap_fig:
            return no_update 
            
        axis_labels = heatmap_fig['data'][0]['x'] 
        new_selection = []
        
        # Scenario 1: Plotly returns explicit points (sometimes happens on single clicks)
        if 'points' in selectedData and len(selectedData['points']) > 0 and 'x' in selectedData['points'][0] and isinstance(selectedData['points'][0]['x'], str):
            x_genes = [p['x'] for p in selectedData['points'] if 'x' in p]
            y_genes = [p['y'] for p in selectedData['points'] if 'y' in p]
            new_selection = list(set(x_genes + y_genes))
            
        # Scenario 2: Plotly returns a bounding box range (Standard for Heatmap Box-Select)
        elif 'range' in selectedData:
            x_range = selectedData['range']['x']
            y_range = selectedData['range']['y']
            
            def get_selected_labels(labels, sel_range):
                res = []
                min_val, max_val = min(sel_range), max(sel_range)
                for i, label in enumerate(labels):
                    # Check if the integer index of the label falls inside the selection box
                    if (i + 0.5) >= min_val and (i - 0.5) <= max_val:
                        res.append(label)
                return res
                
            sel_x = get_selected_labels(axis_labels, x_range)
            sel_y = get_selected_labels(axis_labels, y_range)
            new_selection = list(set(sel_x + sel_y))

        if not new_selection:
            return no_update
            
        return new_selection

    # Renderer Callback (Dropdown -> Small Multiples)   
    @app.callback(
        Output('small-multiples-heatmaps', 'figure'),
        Input('gene-tracker-dropdown', 'value'),
        Input('tracker-time-1', 'value'),
        Input('tracker-time-2', 'value')
    )
    def update_small_multiples(selected_genes, t1, t2):
        if not selected_genes or not t1 or not t2:
            return go.Figure().update_layout(
                annotations=[dict(text="Select genes and timepoints to track rewiring.", showarrow=False, font=dict(color="gray"))],
                xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor='white',
                margin=dict(l=0, r=0, t=30, b=0)
            )

        compare_times = [t1, t2]

        # Explicitly create a 1x2 grid
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=compare_times,
            horizontal_spacing=0.05
        )

        for i, time in enumerate(compare_times):
            # Fetch the pre-clustered full matrix for this timepoint
            full_df = data_loader.tf_correlation_dfs[time] 
            
            # Extract subset while preserving global hierarchical order
            global_order = list(full_df.columns)
            ordered_subset = [g for g in global_order if g in selected_genes]
            
            if not ordered_subset:
                continue

            df_subset = full_df.loc[ordered_subset, ordered_subset]
            col_idx = i + 1
            
            fig.add_trace(go.Heatmap(
                yaxis_autorange='reversed',
                z=df_subset.values, 
                x=df_subset.columns, 
                y=df_subset.index,
                colorscale='RdBu_r', zmid=0, zmin=-0.8, zmax=0.8,
                showscale=(i == 1), # Only show colorbar on the right-most plot
                colorbar=dict(title="Spearman", thickness=10) if (i == 1) else None,
                hovertemplate="TF X: %{x}<br>TF Y: %{y}<br>Corr: %{z:.2f}<extra></extra>"
            ), row=1, col=col_idx)

        fig.update_layout(
            plot_bgcolor='white',
            margin=dict(l=0, r=0, t=30, b=10)
        )
        
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return fig


    # =========================================================================
    # CALLBACKS: TAB 3 (Differential Topology)
    # =========================================================================
    
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
        active_nodes = set() 
        
        nodes = list(circular_pos.keys())
        
        for u, v in itertools.combinations(nodes, 2):
            if u in delta_df.index and v in delta_df.columns:
                delta = delta_df.loc[u, v]
                
                if abs(delta) >= threshold:
                    active_nodes.update([u, v]) 
                    
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
                    hover_customdata.append([u, v]) 

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
            customdata=hover_customdata,
            hovertemplate="%{text}<extra></extra>"
        )

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
        empty_fig = go.Figure().update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text="Click an edge to view<br>metacell expression.", xref="paper", yref="paper", showarrow=False, font=dict(color="gray"))],
            plot_bgcolor='white', margin=dict(l=0, r=0, t=30, b=0)
        )
        if not clickData or not t1 or not t2:
            return empty_fig, empty_fig

        tf_u, tf_v = clickData['points'][0]['customdata']

        def create_scatter(time, tf_x, tf_y):
            df_exp = data_loader.get_metacell_expression(time, [tf_x, tf_y])

            try:
                corr_val = data_loader.tf_correlation_dfs[time].loc[tf_x, tf_y]
            except KeyError:
                corr_val = 0.0 
            
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
                plot_bgcolor='rgba(240,240,240,0.5)' 
            )
            return fig

        fig_t1 = create_scatter(t1, tf_u, tf_v)
        fig_t2 = create_scatter(t2, tf_u, tf_v)

        return fig_t1, fig_t2


    # =========================================================================
    # CALLBACKS: TAB 2 (Targeted Dynamics)
    # =========================================================================

    @app.callback(
        Output('ego-heatmap', 'figure'),
        Input('ego-tf-dropdown', 'value')
    )
    def update_1d_heatmap(tf_name):
        if not tf_name: return go.Figure()
        
        # 1. Fetch data
        ego_data = {t: data_loader.tf_correlation_dfs[t].loc[tf_name] for t in timepoints}
        ego_df = pd.DataFrame(ego_data)
        
        # 2. Remove the target TF itself so it doesn't just show a solid block of 1.0 correlation
        if tf_name in ego_df.index: ego_df = ego_df.drop(index=tf_name)
        
        # 3. Sort by max absolute correlation across all timepoints (NO MORE TOP 15 LIMIT!)
        ego_df = ego_df.loc[ego_df.abs().max(axis=1).sort_values(ascending=False).index]

        # 4. Calculate dynamic height based on the full list of TFs
        plot_height = max(300, len(ego_df) * 20 + 100)

        # 5. Build Heatmap
        fig = go.Figure(go.Heatmap(
            z=ego_df.values, x=ego_df.columns, y=ego_df.index,
            colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
            text=np.round(ego_df.values, 2), texttemplate="%{text}", textfont={"size":10},
            colorbar=dict(title="Spearman", thickness=15),
            hovertemplate="<b>Partner TF:</b> %{y}<br><b>Time:</b> %{x}<br><b>Correlation:</b> %{z:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            height=plot_height, # Apply dynamic height
            margin=dict(l=20, r=20, t=20, b=40), 
            yaxis_autorange='reversed',
            xaxis=dict(title="Timepoints", showgrid=False, tickmode='array', tickvals=timepoints),
            yaxis=dict(title="Partner TFs", showgrid=False),
            plot_bgcolor='white'
        )
        return fig

    @app.callback(
        Output('split-streamgraph', 'figure'), 
        Input('gene-picker', 'value')
    )
    def update_temporal_dot_plot(gene_name): # Kept the name the same so it plugs right in
        if not gene_name: return go.Figure()

        df_coef = data_loader.get_in_edges_dataframe(gene_name, 'coefficient')
        if df_coef.empty: return go.Figure()
        
        if 'avg' in df_coef.columns: df_coef = df_coef.drop(columns=['avg'])
        
        # 1. FIX THE X-AXIS ORDER: explicitly subset columns in chronological order
        ordered_timepoints = [t for t in timepoints if t in df_coef.columns]
        df_coef = df_coef[ordered_timepoints]
        
        # 2. FILTER NOISE: Keep TFs with at least one significant timepoint
        noise_threshold = 1e-2
        df_coef = df_coef.loc[(df_coef.abs() > noise_threshold).any(axis=1)]

        # Empty state fallback if all edges were below the noise threshold
        if df_coef.empty:
            return go.Figure().update_layout(
                annotations=[dict(text="No significant upstream regulators found above noise threshold.", showarrow=False, font=dict(color="gray"))],
                xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor='white'
            )

        df_coef = df_coef.loc[df_coef.abs().max(axis=1).sort_values(ascending=False).index]

        plot_height = max(300, len(df_coef) * 20 + 100)

        # 3. CREATE THE HEATMAP
        fig = go.Figure(go.Heatmap(
            z=df_coef.values, 
            x=df_coef.columns, 
            y=df_coef.index,
            colorscale='RdBu_r', 
            zmid=0, 
            zmin=-global_max_coef, 
            zmax=global_max_coef,
            colorbar=dict(title="Ridge<br>Coefficient", thickness=15),
            # Optional but highly recommended: overlay the exact numbers on the cells
            text=np.round(df_coef.values, 3), 
            texttemplate="%{text}",
            hovertemplate="<b>TF:</b> %{y}<br><b>Time:</b> %{x}<br><b>Coefficient:</b> %{z:.3f}<extra></extra>"
        ))

        fig.update_layout(
            height=plot_height,
            margin=dict(l=20, r=20, t=40, b=40),
            xaxis=dict(title="Timepoints", showgrid=False, tickmode='array', tickvals=ordered_timepoints),
            yaxis=dict(title="Upstream TFs", showgrid=False, autorange='reversed'), 
            plot_bgcolor='white',
        )
        
        return fig

    @app.callback(
        Output('downstream-heatmap', 'figure'),
        Input('downstream-tf-picker', 'value')
    )
    def update_downstream_heatmap(tf_name):
        if not tf_name: return go.Figure()

        # 1. Fetch the data using your native method
        try:
            # Assuming the edge attribute is literally named 'correlation'
            df_targets = data_loader.get_out_edges_dataframe(tf_name, 'correlation', threshold=0.1)
        except Exception:
            df_targets = pd.DataFrame() # Safety catch if TF isn't in the network

        # Empty state fallback
        if df_targets.empty:
            return go.Figure().update_layout(
                annotations=[dict(text=f"No downstream targets found for {tf_name} above threshold.", showarrow=False, font=dict(color="gray"))],
                xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor='white'
            )

        # 2. Clean up for plotting (Remove the 'avg' column and enforce chronology)
        if 'avg' in df_targets.columns:
            df_targets = df_targets.drop(columns=['avg'])
            
        ordered_timepoints = [t for t in timepoints if t in df_targets.columns]
        df_targets = df_targets[ordered_timepoints]

        df_targets = df_targets.loc[df_targets.abs().max(axis=1).sort_values(ascending=False).index]

        # 3. Calculate dynamic height (20px per gene row + 100px for margins/labels)
        plot_height = max(300, len(df_targets) * 20 + 100)

        # 4. Build the Heatmap
        fig = go.Figure(go.Heatmap(
            z=df_targets.values,
            x=df_targets.columns,
            y=df_targets.index,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1, # Spearman natively scales from -1 to 1
            zmax=1,
            colorbar=dict(title="Spearman<br>Correlation", thickness=15),
            text=np.round(df_targets.values, 2),
            texttemplate="%{text}",
            hovertemplate="<b>Target Gene:</b> %{y}<br><b>Time:</b> %{x}<br><b>Correlation:</b> %{z:.2f}<extra></extra>"
        ))

        fig.update_layout(
            height=plot_height, 
            margin=dict(l=20, r=20, t=40, b=40),
            xaxis=dict(title="Timepoints", showgrid=False, tickmode='array', tickvals=ordered_timepoints),
            yaxis=dict(title="Target Genes", showgrid=False, autorange='reversed'),
            plot_bgcolor='white',
        )

        return fig

    app.run(debug=True)