import itertools

import dash_bootstrap_components as dbc
import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from scipy.cluster.hierarchy import fcluster

from tangerine.visualise.data_loader import DataLoader
from tangerine.visualise.utils import order_dataframe_by_linkage


def make_app_layout(app, tf_list, gene_list, timepoints, global_max_coef):
    t0 = timepoints[0]
    t1 = timepoints[-1] if len(timepoints) > 1 else timepoints[0]
    tf_chosen = tf_list[0] if tf_list else None
    gene_chosen = gene_list[0] if gene_list else None

    ridge_max = float(np.round(global_max_coef, 3))
    ridge_step = float(np.round(ridge_max / 50, 4))  # 50 discrete steps
    if ridge_step == 0:
        ridge_step = 0.001
    ridge_default = float(
        np.round(ridge_max * 0.05, 3)
    )  # Default to filtering out the bottom 5%

    # Create clean marks for the slider UI
    ridge_marks = {
        0: "0",
        ridge_max / 2: f"{ridge_max / 2:.2f}",
        ridge_max: f"{ridge_max:.2f}",
    }

    manuscript_config = {
        "toImageButtonOptions": {
            "format": "png",
            "scale": 4,
            "filename": "tangerine_export",
            # "width": 3000,  
            # "height": 900,
        }
    }

    tf_list.sort()
    gene_list.sort()

    app.layout = dbc.Container(
        [
            # --- HEADER ---
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H2(
                                "🍊 Tangerine: Dynamic Gene Regulatory Network Explorer",
                                className="text-primary",
                            ),
                            html.P(
                                "Explore time-varying TF co-regulation and regulatory network topology.",
                                className="mb-0",
                            ),
                        ]
                    )
                ],
                className="mb-4 mt-3 shadow-sm",
            ),
            # --- TABS ---
            dbc.Tabs(
                [
                    # ==========================================================
                    # TAB 1: GLOBAL TOPOLOGY & MODULES
                    # ==========================================================
                    # ==========================================================
                    # TAB 1: GLOBAL TOPOLOGY & MODULES
                    # ==========================================================
                    dbc.Tab(
                        label="Global Topology & Modules",
                        tab_id="tab-1",
                        active_tab_style={"backgroundColor": "#5c5c5c", "color": "white", "fontWeight": "bold"},
                        children=[
                            # --------------------------------------------------
                            # TOP ROW: Global Heatmap (Left) & Sankey (Right)
                            # --------------------------------------------------
                            dbc.Row([
                                # LEFT: Square Global Heatmap (Width 5)
                                dbc.Col([
                                    html.H5("Global TF Correlation", className="mt-3 fw-bold"),
                                    html.P("Select timepoint. Box-select to extract TFs.", className="text-muted small mb-2"),
                                    html.Div(
                                        dcc.Slider(
                                            0,
                                            len(timepoints) - 1,
                                            step=1,
                                            marks={
                                                i: {
                                                    "label": t,
                                                    "style": {
                                                        # Alternates margin based on whether the index is even or odd
                                                        "marginTop": "5px" if i % 2 == 0 else "25px",
                                                        "fontSize": "10px",
                                                        "whiteSpace": "nowrap",
                                                    },
                                                }
                                                for i, t in enumerate(timepoints)
                                            },
                                            value=0,
                                            id="heatmap-time-slider",
                                            className="mb-2",
                                        ),
                                        style={"paddingBottom": "40px"} # Room for the lower row of text
                                    ),
                                    dcc.Graph(
                                        id="correlation-heatmap",
                                        style={"height": "500px"}, # Forced to be more square
                                        config=manuscript_config,
                                    ),
                                ], width=5, className="border-end pe-4"),
                                
                                # RIGHT: Alluvial/Sankey Plot (Width 7)
                                dbc.Col([
                                    html.H5("Module Evolution", className="mt-3 fw-bold"),
                                    html.P("Tracks cluster membership of TFs across time.", className="text-muted small mb-2"),
                                    html.Div([
                                        html.Label("Granularity (\u0394k):", className="small text-muted mb-0 me-3 text-nowrap"), 
                                        html.Div(
                                            dcc.Slider(
                                                min=-2, max=5, step=1, value=0, id="sankey-granularity-slider",
                                                marks={-2: "-2", 0: "0", 5: "+5"}
                                            ),
                                            style={"flex": "1"} 
                                        )
                                    ], className="mb-3 d-flex align-items-center"),
                                    html.Div(
                                        dcc.Graph(id="alluvial-plot", style={"height": "450px"}, config=manuscript_config,),
                                        style={"overflowX": "auto", "width": "100%"},
                                    ),
                                ], width=7, className="ps-4"),
                            ], className="mb-4 mt-2"),

                            html.Hr(className="mb-4 text-muted"), # Visual separator

                            # --------------------------------------------------
                            # MIDDLE BAR: Dropdown Control & Clipboard
                            # --------------------------------------------------
                            dbc.Row([
                                dbc.Col([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("Toggle Selected TFs:", className="mt-3 fw-bold"),
                                            html.P("Explore in detail selected TFs. Ordering is anchored to selected timepoint.", className="text-muted small mb-2"),
                                    ]),
                                        dbc.Col([
                                            html.Span("Copy List", className="text-muted small me-2"),
                                            dcc.Clipboard(id="copy-clipboard", style={"display": "inline-block", "fontSize": "18px", "color": "#158cba", "cursor": "pointer"}),
                                        ], width="auto", className="text-end")
                                    ], justify="between"),
                                    dcc.Dropdown(
                                        id="gene-tracker-dropdown",
                                        options=[{"label": g, "value": g} for g in tf_list],
                                        value=[], multi=True,
                                        placeholder="Box-select on heatmap to import, or type here...",
                                        className="shadow-sm"
                                    ),
                                ], width=12)
                            ], className="mb-4"),

                            # --------------------------------------------------
                            # BOTTOM ROW: 3 Sub-Heatmaps
                            # --------------------------------------------------
                            dbc.Row([
                                # Plot 1: Past/Custom
                                dbc.Col([
                                    dbc.Select(
                                        options=[{"label": f"{t}", "value": t} for t in timepoints],
                                        value=timepoints[0], id="tracker-time-1", className="mb-2 shadow-sm"
                                    ),
                                    dcc.Graph(id="sub-heatmap-1", style={"height": "400px"}, config=manuscript_config,)
                                ], width=4),
                                
                                # Plot 2: Anchor (Fixed to Global Slider)
                                dbc.Col([
                                    html.Div(
                                        "Selected Timepoint (Anchor)", 
                                        className="text-center fw-bold text-black bg-secondary rounded py-1 mb-2 shadow-sm",
                                        id="tracker-time-mid-label"
                                    ),
                                    dcc.Graph(id="sub-heatmap-mid", style={"height": "400px"}, config=manuscript_config,)
                                ], width=4),
                                
                                # Plot 3: Future/Custom
                                dbc.Col([
                                    dbc.Select(
                                        options=[{"label": f"{t}", "value": t} for t in timepoints],
                                        value=timepoints[-1], id="tracker-time-2", className="mb-2 shadow-sm"
                                    ),
                                    dcc.Graph(id="sub-heatmap-2", style={"height": "400px"}, config=manuscript_config,)
                                ], width=4),
                            ], className="mb-5")
                        ],
                    ),
                    # ==========================================================
                    # TAB 2: TARGETED DYNAMICS
                    # ==========================================================
                    dbc.Tab(
                        label="Targeted Dynamics",
                        tab_id="tab-2",
                        active_tab_style={"backgroundColor": "#5c5c5c", "color": "white", "fontWeight": "bold"},
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4(
                                                "Targeted Search & Dynamics",
                                                className="mt-4",
                                            ),
                                            html.P(
                                                "Expand a panel below to explore specific gene or TF regulatory relationships.",
                                                className="text-muted mb-4",
                                            ),
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                        html.Strong(
                                                            id="violin-plot-title",
                                                            children="Baseline Expression",
                                                        )
                                                    ),
                                                    dbc.CardBody(
                                                        [
                                                            dcc.Graph(
                                                                id="global-expression-violin",
                                                                style={
                                                                    "height": "200px"
                                                                },
                                                                config=manuscript_config,
                                                            )
                                                        ],
                                                        className="py-2",
                                                    ),
                                                ],
                                                className="mb-4 shadow-sm border-0",
                                            ),
                                            # The Accordion automatically manages expanding/collapsing
                                            dbc.Accordion(
                                                [
                                                    # PANEL 3: Co-regulation (Ego Network - SPEARMAN)
                                                    dbc.AccordionItem(
                                                        [
                                                            html.P(
                                                                "How does a specific TF's relationship with all other TFs evolve over time?",
                                                                className="text-muted small",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        dbc.InputGroup(
                                                                            [
                                                                                dbc.InputGroupText(
                                                                                    "Target TF:"
                                                                                ),
                                                                                dbc.Select(
                                                                                    options=[
                                                                                        {
                                                                                            "label": tf,
                                                                                            "value": tf,
                                                                                        }
                                                                                        for tf in tf_list
                                                                                    ],
                                                                                    value=tf_chosen,
                                                                                    id="ego-tf-dropdown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        width=4,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Correlation Threshold:",
                                                                                className="small text-muted mb-0",
                                                                            ),
                                                                            dcc.Slider(
                                                                                0,
                                                                                0.8,
                                                                                step=0.05,
                                                                                value=0.1,
                                                                                id="ego-corr-slider",
                                                                                marks={
                                                                                    0: "0",
                                                                                    0.4: "0.4",
                                                                                    0.8: "0.8",
                                                                                },
                                                                            ),
                                                                        ],
                                                                        width=4,
                                                                    ),
                                                                ],
                                                                className="mb-3 align-items-center",
                                                            ),
                                                            html.Div(
                                                                dcc.Graph(
                                                                    id="ego-heatmap",
                                                                    config=manuscript_config,
                                                                ),
                                                                style={
                                                                    "overflowY": "auto",
                                                                    "maxHeight": "600px",
                                                                    "border": "1px solid #f0f0f0",
                                                                },
                                                            ),
                                                        ],
                                                        title="TF Co-regulation Dynamics",
                                                    ),
                                                    # PANEL 2: Downstream (TF Targets - SPEARMAN)
                                                    dbc.AccordionItem(
                                                        [
                                                            html.P(
                                                                "Explore all genes regulated by a specific TF, colored by Spearman correlation.",
                                                                className="text-muted small",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        dbc.InputGroup(
                                                                            [
                                                                                dbc.InputGroupText(
                                                                                    "Regulator TF:"
                                                                                ),
                                                                                dbc.Select(
                                                                                    options=[
                                                                                        {
                                                                                            "label": tf,
                                                                                            "value": tf,
                                                                                        }
                                                                                        for tf in tf_list
                                                                                    ],
                                                                                    value=tf_chosen,
                                                                                    id="downstream-tf-picker",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        width=4,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Correlation Threshold:",
                                                                                className="small text-muted mb-0",
                                                                            ),
                                                                            dcc.Slider(
                                                                                0,
                                                                                0.8,
                                                                                step=0.05,
                                                                                value=0.1,
                                                                                id="downstream-corr-slider",
                                                                                marks={
                                                                                    0: "0",
                                                                                    0.4: "0.4",
                                                                                    0.8: "0.8",
                                                                                },
                                                                            ),
                                                                        ],
                                                                        width=4,
                                                                    ),
                                                                ],
                                                                className="mb-3 align-items-center",
                                                            ),
                                                            html.Div(
                                                                dcc.Graph(
                                                                    id="downstream-heatmap",
                                                                    config=manuscript_config,
                                                                ),
                                                                style={
                                                                    "overflowY": "auto",
                                                                    "maxHeight": "600px",
                                                                    "border": "1px solid #f0f0f0",
                                                                },
                                                            ),
                                                        ],
                                                        title="Downstream Targets (What does this TF drive?)",
                                                    ),
                                                    # PANEL 1: Upstream (Target Gene - RIDGE)
                                                    dbc.AccordionItem(
                                                        [
                                                            html.P(
                                                                "Total upstream influence on a target gene over time (Ridge Regression).",
                                                                className="text-muted small",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        dbc.InputGroup(
                                                                            [
                                                                                dbc.InputGroupText(
                                                                                    "Target Gene:"
                                                                                ),
                                                                                dbc.Select(
                                                                                    options=[
                                                                                        {
                                                                                            "label": g,
                                                                                            "value": g,
                                                                                        }
                                                                                        for g in gene_list
                                                                                    ],
                                                                                    value=gene_chosen,
                                                                                    id="gene-picker",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        width=4,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Ridge Noise Threshold:",
                                                                                className="small text-muted mb-0",
                                                                            ),
                                                                            dcc.Slider(
                                                                                min=0,
                                                                                max=ridge_max,
                                                                                step=ridge_step,
                                                                                value=ridge_default,
                                                                                id="ridge-threshold-slider",
                                                                                marks=ridge_marks,
                                                                            ),
                                                                        ],
                                                                        width=4,
                                                                    ),
                                                                ],
                                                                className="mb-3 align-items-center",
                                                            ),
                                                            html.Div(
                                                                dcc.Graph(
                                                                    id="split-streamgraph",
                                                                    config=manuscript_config,
                                                                ),
                                                                style={
                                                                    "overflowY": "auto",
                                                                    "maxHeight": "600px",
                                                                    "border": "1px solid #f0f0f0",
                                                                },
                                                            ),
                                                        ],
                                                        title="Upstream Regulators (Who drives this gene?)",
                                                    ),
                                                ],
                                                start_collapsed=False,
                                                always_open=False,
                                                active_item="item-0",
                                            ),  # item-0 opens the first panel by default
                                        ],
                                        width=12,
                                    )
                                ],
                                className="mb-5",
                            )
                        ],
                    ),
                    # ==========================================================
                    # TAB 3: DIFFERENTIAL TOPOLOGY (Moved from Tab 1)
                    # ==========================================================
                    # ==========================================================
                    # TAB 3: DIFFERENTIAL TOPOLOGY
                    # ==========================================================
                    dbc.Tab(
                        label="Differential Topology",
                        tab_id="tab-3",
                        active_tab_style={"backgroundColor": "#5c5c5c", "color": "white", "fontWeight": "bold"},
                        children=[
                            dbc.Row(
                                [
                                    # LEFT SECTION: Controls + Circular Graph (Width 8)
                                    dbc.Col(
                                        [
                                            html.H4(
                                                "Differential Network Topology",
                                                className="mt-4",
                                            ),
                                            html.P(
                                                "Compare two timepoints. Edges show change in Spearman correlation.",
                                                className="text-muted",
                                            ),
                                            dbc.Row(
                                                [
                                                    # Controls Sidebar Panel
                                                    dbc.Col(
                                                        [
                                                            dbc.Card(
                                                                [
                                                                    dbc.CardBody(
                                                                        [
                                                                            html.Label(
                                                                                "Compare Timepoints:",
                                                                                className="fw-bold small",
                                                                            ),
                                                                            dbc.InputGroup(
                                                                                [
                                                                                    dbc.Select(
                                                                                        options=[
                                                                                            {
                                                                                                "label": t,
                                                                                                "value": t,
                                                                                            }
                                                                                            for t in timepoints
                                                                                        ],
                                                                                        value=t0,
                                                                                        id="diff-time-1",
                                                                                        style={"minWidth": "90px"}
                                                                                    ),
                                                                                    dbc.InputGroupText(
                                                                                        "vs"
                                                                                    ),
                                                                                    dbc.Select(
                                                                                        options=[
                                                                                            {
                                                                                                "label": t,
                                                                                                "value": t,
                                                                                            }
                                                                                            for t in timepoints
                                                                                        ],
                                                                                        value=t1,
                                                                                        id="diff-time-2",
                                                                                        style={"minWidth": "90px"}
                                                                                    ),
                                                                                ],
                                                                                className="mb-4",
                                                                            ),
                                                                            html.Label(
                                                                                "Δ Correlation Threshold:",
                                                                                className="fw-bold small mb-1",
                                                                            ),
                                                                            html.P(
                                                                                "Filter noise from the network.",
                                                                                className="text-muted small mb-3",
                                                                            ),
                                                                            dcc.Slider(
                                                                                0.1,
                                                                                1.0,
                                                                                step=0.05,
                                                                                value=0.75,
                                                                                id="delta-threshold",
                                                                                marks={
                                                                                    0.2: "0.2",
                                                                                    0.5: "0.5",
                                                                                    0.8: "0.8",
                                                                                },  # Simplified marks for a cleaner look
                                                                            ),
                                                                        ],
                                                                        className="p-3",
                                                                    )
                                                                ],
                                                                className="shadow-sm border-0 bg-light mt-4",
                                                            )
                                                        ],
                                                        width=4,
                                                    ),
                                                    # Circular Layout Graph
                                                    dbc.Col(
                                                        [
                                                            dcc.Graph(
                                                                id="differential-circular-graph",
                                                                style={
                                                                    "height": "600px"
                                                                },
                                                                config=manuscript_config,
                                                            )
                                                        ],
                                                        width=8,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        width=8,
                                        className="border-end pe-4",
                                    ),
                                    # RIGHT SECTION: Stacked Scatterplots (Width 4)
                                    dbc.Col(
                                        [
                                            html.H5(
                                                "Raw Metacell Expression",
                                                className="mt-4",
                                            ),
                                            html.P(
                                                "Click an edge in the network to view.",
                                                className="text-muted small",
                                            ),
                                            # Stacked vertically for the sidebar layout
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="scatter-t1",
                                                            style={"height": "320px"},
                                                            config=manuscript_config,
                                                        ),
                                                        width=12,
                                                        className="mb-3",
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="scatter-t2",
                                                            style={"height": "320px"},
                                                            config=manuscript_config,
                                                        ),
                                                        width=12,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        width=4,
                                        className="ps-4",
                                    ),
                                ],
                                className="mb-5 mt-2",
                            )
                        ],
                    ),
                ],
                id="tabs",
                active_tab="tab-1",
                className="mt-3",
            ),
        ],
        fluid=True,
        className="dbc px-5",
    )


def run_app(timepoints, base_path):
    data_loader = DataLoader(timepoints, base_path)

    # Calculate global max coefficient for consistent dot plot colors
    global_max_coef = 0.0
    for time in timepoints:
        for u, v, data in data_loader.networks[time].edges(data=True):
            val = abs(data.get("coefficient", 0))
            if val > global_max_coef:
                global_max_coef = val

    if global_max_coef == 0:
        global_max_coef = 1.0  # Fallback safety

    t0 = timepoints[0]
    tf_list = data_loader.tf_list
    gene_list = []
    for node in data_loader.networks[t0]:
        if len(data_loader.networks[t0].in_edges(node)) > 0:
            gene_list.append(node)

    dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

    app = Dash(__name__, external_stylesheets=[dbc.themes.LUMEN, dbc_css])
    make_app_layout(app, tf_list, gene_list, timepoints, global_max_coef)

    vmin, vmax = -0.75, 0.75

    # =========================================================================
    # CALLBACKS: TAB 1 (Global Topology & Small Multiples)
    # =========================================================================

    @app.callback(
        Output("correlation-heatmap", "figure"), Input("heatmap-time-slider", "value")
    )
    def update_heatmap(time_idx):
        time = timepoints[time_idx]
        df = data_loader.tf_correlation_dfs[time]

        fig = go.Figure(
            go.Heatmap(
                z=df.values,
                x=df.columns,
                y=df.index,
                colorscale="RdBu_r",
                zmid=0,
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title="Spearman"),
                hovertemplate="TF X: %{x}<br>TF Y: %{y}<br>Corr: %{z:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_showticklabels=False,
            yaxis_showticklabels=False,
            yaxis_autorange="reversed",
            dragmode="select",
            plot_bgcolor="white",
        )
        return fig

    @app.callback(
        Output("alluvial-plot", "figure"),
        Output("copy-clipboard", "content"), # Keep only this and the figure
        Input("correlation-heatmap", "selectedData"),
        Input("sankey-granularity-slider", "value"),
        State("correlation-heatmap", "figure"),
    )
    def update_alluvial_and_chips(selectedData, granularity_offset, heatmap_fig):
        selected_tfs = []

        # 1. Handle Selection Logic
        if selectedData and heatmap_fig and "data" in heatmap_fig:
            axis_labels = heatmap_fig["data"][0]["x"]

            if (
                "points" in selectedData
                and len(selectedData["points"]) > 0
                and "x" in selectedData["points"][0]
                and isinstance(selectedData["points"][0]["x"], str)
            ):
                selected_tfs = list(
                    set(
                        [p["x"] for p in selectedData["points"]]
                        + [p["y"] for p in selectedData["points"]]
                    )
                )

            elif "range" in selectedData:
                x_range = selectedData["range"]["x"]
                y_range = selectedData["range"]["y"]

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

        # 2. Extract Sankey Nodes (Unique Clusters)
        df_clusters = data_loader.tf_louvain.copy()
        time_cols = [t for t in timepoints if t in df_clusters.columns]

        for t in time_cols:
            Z = data_loader.linkage_matrices[t]
            optimal_k = data_loader.baseline_k[t]
            new_k = max(2, optimal_k + granularity_offset)

            labels = fcluster(Z, t=new_k, criterion="maxclust")

            tf_names = data_loader.tf_correlation_dfs[t].index
            label_series = pd.Series(
                [f"{t}-{int(lbl)}" for lbl in labels], index=tf_names
            )

            df_clusters[t] = label_series

        unique_nodes = []
        for t in time_cols:
            unique_nodes.extend(df_clusters[t].dropna().unique())

        node_idx_map = {node: i for i, node in enumerate(unique_nodes)}

        # 3. Calculate Node Colors (Average Intra-Cluster Correlation)
        node_colors = []
        cmap = mpl.cm.get_cmap("RdBu_r")
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        for node in unique_nodes:
            t_node = next((t for t in time_cols if str(node).startswith(t)), None)

            if t_node:
                tfs_in_cluster = df_clusters[df_clusters[t_node] == node].index.tolist()
                corr_df = data_loader.tf_correlation_dfs[t_node]
                valid_tfs = [tf for tf in tfs_in_cluster if tf in corr_df.index]

                if len(valid_tfs) > 1:
                    sub_corr = corr_df.loc[valid_tfs, valid_tfs].values
                    idx = np.triu_indices_from(sub_corr, k=1)
                    avg_c = np.nanmean(sub_corr[idx])
                else:
                    avg_c = 1.0

                hex_color = mpl.colors.to_hex(cmap(norm(avg_c)))
                node_colors.append(hex_color)
            else:
                node_colors.append("#cccccc")

        # 4. Extract Links, Split Paths, and Build Tooltips
        sources, targets, values, link_colors, link_customdata = [], [], [], [], []

        for i in range(len(time_cols) - 1):
            t1 = time_cols[i]
            t2 = time_cols[i + 1]

            # Map: (source_cluster, target_cluster, is_selected) -> List of TFs
            transitions = {}
            for tf in df_clusters.index:
                src = df_clusters.loc[tf, t1]
                tgt = df_clusters.loc[tf, t2]

                if pd.isna(src) or pd.isna(tgt):
                    continue

                is_sel = tf in selected_tfs
                key = (src, tgt, is_sel)

                if key not in transitions:
                    transitions[key] = []
                transitions[key].append(tf)

            # Sort transitions so highlighted arcs (is_sel=True) are drawn on top
            sorted_transitions = sorted(
                transitions.items(), key=lambda item: item[0][2]
            )

            for (src, tgt, is_sel), tfs_in_transition in sorted_transitions:
                sources.append(node_idx_map[src])
                targets.append(node_idx_map[tgt])
                values.append(len(tfs_in_transition))

                link_colors.append(
                    "rgba(74, 74, 74, 0.7)" if is_sel else "rgba(240, 240, 240, 0.5)"
                )

                # Build the Tooltip HTML String
                if is_sel:
                    # Truncate the list if there are more than 8 genes
                    if len(tfs_in_transition) > 8:
                        gene_str = (
                            ", ".join(tfs_in_transition[:8])
                            + f" ... (+{len(tfs_in_transition) - 8} more)"
                        )
                    else:
                        gene_str = ", ".join(tfs_in_transition)

                    hover_html = f"<b>Highlighted Transition</b><br>{len(tfs_in_transition)} Genes: {gene_str}"
                else:
                    hover_html = f"<b>Background Transition</b><br>{len(tfs_in_transition)} Genes"

                link_customdata.append(hover_html)

        # 5. Build the Sankey Figure
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="white", width=0.5),
                        label=unique_nodes,
                        color=node_colors,
                        hoverinfo="none",  # Keep node tooltips disabled for cleanliness
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                        customdata=link_customdata,  # Inject the pre-formatted HTML strings
                        hovertemplate="%{customdata}<extra></extra>",  # <extra></extra> hides the secondary trace box
                    ),
                )
            ]
        )

        # Calculate dynamic width (150px per timepoint, min 800px)
        dynamic_width = max(800, len(time_cols) * 130)

        # Apply the width to the layout
        fig.update_layout(
            width=dynamic_width,
            margin=dict(l=20, r=40, t=20, b=20),
            plot_bgcolor="white",
        )

        tf_string = "\n".join(selected_tfs) if selected_tfs else ""
        return fig, tf_string

    # Importer Callback (Heatmap -> Dropdown)
    @app.callback(
        Output("gene-tracker-dropdown", "value"),
        Input("correlation-heatmap", "selectedData"),
        State("correlation-heatmap", "figure"),
    )
    def import_genes_from_heatmap(selectedData, heatmap_fig):
        if not selectedData or not heatmap_fig or "data" not in heatmap_fig:
            return no_update

        axis_labels = heatmap_fig["data"][0]["x"]
        new_selection = []

        # Scenario 1: Plotly returns explicit points
        if (
            "points" in selectedData
            and len(selectedData["points"]) > 0
            and "x" in selectedData["points"][0]
            and isinstance(selectedData["points"][0]["x"], str)
        ):
            x_genes = [p["x"] for p in selectedData["points"] if "x" in p]
            y_genes = [p["y"] for p in selectedData["points"] if "y" in p]
            new_selection = list(set(x_genes + y_genes))

        # Scenario 2: Plotly returns a bounding box range (Standard for Box-Select)
        elif "range" in selectedData:
            x_range = selectedData["range"]["x"]
            y_range = selectedData["range"]["y"]

            def get_selected_labels(labels, sel_range):
                res = []
                min_val, max_val = min(sel_range), max(sel_range)
                for i, label in enumerate(labels):
                    if (i + 0.5) >= min_val and (i - 0.5) <= max_val:
                        res.append(label)
                return res

            sel_x = get_selected_labels(axis_labels, x_range)
            sel_y = get_selected_labels(axis_labels, y_range)
            new_selection = list(set(sel_x + sel_y))

        if not new_selection:
            return no_update

        return new_selection

    @app.callback(
        Output("sub-heatmap-1", "figure"),
        Output("sub-heatmap-mid", "figure"),
        Output("sub-heatmap-2", "figure"),
        Output("tracker-time-mid-label", "children"),
        Input("gene-tracker-dropdown", "value"),
        Input("tracker-time-1", "value"),
        Input("heatmap-time-slider", "value"), 
        Input("tracker-time-2", "value"),
    )
    def update_3_sub_heatmaps(selected_genes, t1, slider_idx, t2):
        t_mid = timepoints[slider_idx]
        mid_label = f"Anchor: {t_mid}"
        
        empty_fig = go.Figure().update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white")

        if not selected_genes or not t1 or not t2:
            return empty_fig, empty_fig, empty_fig, mid_label

        # 1. Get the data for the Anchor (Middle) timepoint
        full_df_mid = data_loader.tf_correlation_dfs[t_mid]
        
        # Safely extract the genes that exist in the matrix
        valid_genes = [g for g in selected_genes if g in full_df_mid.index]
        if len(valid_genes) < 2:
            return empty_fig, empty_fig, empty_fig, mid_label

        df_mid_subset = full_df_mid.loc[valid_genes, valid_genes]
        
        # 2. Cluster the middle subset to get the master ordering
        # (Using the order_dataframe_by_linkage util you already have)
        df_mid_ordered = order_dataframe_by_linkage(df_mid_subset)
        master_order = list(df_mid_ordered.index)

        # 3. Helper function to plot a single heatmap
        def build_sub_heatmap(t_target):
            df_target = data_loader.tf_correlation_dfs[t_target]
            
            # Extract and force the exact master order
            df_plot = df_target.loc[master_order, master_order]
            
            fig = go.Figure(go.Heatmap(
                z=df_plot.values, x=df_plot.columns, y=df_plot.index,
                colorscale="RdBu_r", zmid=0, zmin=vmin, zmax=vmax,
                showscale=True, 
                hovertemplate="X: %{x}<br>Y: %{y}<br>Corr: %{z:.2f}<extra></extra>",
            ))
            
            fig.update_layout(
                plot_bgcolor="white", 
                margin=dict(l=0, r=0, t=10, b=10),
                xaxis=dict(showticklabels=False), 
                yaxis=dict(showticklabels=False, autorange="reversed")
            )
            return fig

        # 4. Build all three figures
        fig1 = build_sub_heatmap(t1)
        fig_mid = build_sub_heatmap(t_mid)
        fig2 = build_sub_heatmap(t2)

        return fig1, fig_mid, fig2, mid_label

    # =========================================================================
    # CALLBACKS: TAB 3 (Differential Topology)
    # =========================================================================

    @app.callback(
        Output("differential-circular-graph", "figure"),
        Input("diff-time-1", "value"),
        Input("diff-time-2", "value"),
        Input("delta-threshold", "value"),
    )
    def update_diff_graph(t1, t2, threshold):
        if not t1 or not t2:
            return go.Figure()

        df1 = data_loader.tf_correlation_dfs[t1]
        df2 = data_loader.tf_correlation_dfs[t2]
        delta_df = df2 - df1
        delta_df = delta_df.fillna(0)

        pos_edge_x, pos_edge_y = [], []
        neg_edge_x, neg_edge_y = [], []

        hover_x, hover_y, hover_text, hover_customdata = [], [], [], []
        active_nodes = set()

        # Extract the saved consensus coordinates directly from the GML data 
        base_network = data_loader.networks[t1]
        node_pos = {n: (data.get('x', 0), data.get('y', 0)) for n, data in base_network.nodes(data=True)}
        
        # Only iterate over nodes that actually have coordinates saved
        nodes = list(node_pos.keys())
        # -------------------------------------------------------------------------------------

        for u, v in itertools.combinations(nodes, 2):
            if u in delta_df.index and v in delta_df.columns:
                delta = delta_df.loc[u, v]

                if abs(delta) >= threshold:
                    active_nodes.update([u, v])

                    # Look up the dynamic coordinates instead of the hardcoded global ones
                    x0, y0 = node_pos[u]
                    x1, y1 = node_pos[v]

                    if delta > 0:
                        pos_edge_x.extend([x0, x1, None])
                        pos_edge_y.extend([y0, y1, None])
                    else:
                        neg_edge_x.extend([x0, x1, None])
                        neg_edge_y.extend([y0, y1, None])

                    hover_x.append((x0 + x1) / 2)
                    hover_y.append((y0 + y1) / 2)
                    hover_text.append(
                        f"<b>{u} ↔ {v}</b><br>Δ Corr: {delta:.3f}<br><i>Click for expression</i>"
                    )
                    hover_customdata.append([u, v])

        pos_edge_trace = go.Scatter(
            x=pos_edge_x,
            y=pos_edge_y,
            mode="lines",
            line=dict(width=1.5, color="rgba(231, 76, 60, 0.7)"),
            hoverinfo="none",
        )

        neg_edge_trace = go.Scatter(
            x=neg_edge_x,
            y=neg_edge_y,
            mode="lines",
            line=dict(width=1.5, color="rgba(52, 152, 219, 0.7)"),
            hoverinfo="none",
        )

        edge_hover_trace = go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(size=12, color="rgba(0,0,0,0)"),
            text=hover_text,
            customdata=hover_customdata,
            hovertemplate="%{text}<extra></extra>",
        )

        node_x = [node_pos[n][0] for n in active_nodes]
        node_y = [node_pos[n][1] for n in active_nodes]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=8, color="#5c5c5c", line=dict(width=1, color="white")),
            text=list(active_nodes),
            textposition="top center",
            textfont=dict(size=14, color="#222222"),
            hoverinfo="none",
        )

        fig = go.Figure(
            data=[pos_edge_trace, neg_edge_trace, edge_hover_trace, node_trace]
        )
        fig.update_layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=40, l=40, r=40, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        )
        return fig

    @app.callback(
        Output("scatter-t1", "figure"),
        Output("scatter-t2", "figure"),
        Input("differential-circular-graph", "clickData"),
        State("diff-time-1", "value"),
        State("diff-time-2", "value"),
    )
    def update_scatterplots(clickData, t1, t2):
        empty_fig = go.Figure().update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text="Click an edge to view<br>metacell expression.",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color="gray"),
                )
            ],
            plot_bgcolor="white",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        if not clickData or not t1 or not t2:
            return empty_fig, empty_fig

        tf_u, tf_v = clickData["points"][0]["customdata"]

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

            fig = go.Figure(
                go.Scatter(
                    x=exp_x,
                    y=exp_y,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="#5c5c5c",
                        opacity=1,
                        line=dict(width=0.5, color="white"),
                    ),
                )
            )
            fig.update_layout(
                title=dict(
                    text=f"<b>{time}</b> (Spearman ρ = {corr_val:.2f})",
                    font=dict(size=12),
                    x=0.5,
                    xanchor="center",
                ),
                xaxis_title=dict(text=tf_x, font=dict(size=10)),
                yaxis_title=dict(text=tf_y, font=dict(size=10)),
                margin=dict(l=30, r=10, t=30, b=30),
                plot_bgcolor="rgba(240,240,240,0.5)",
            )
            return fig

        fig_t1 = create_scatter(t1, tf_u, tf_v)
        fig_t2 = create_scatter(t2, tf_u, tf_v)

        return fig_t1, fig_t2

    # =========================================================================
    # CALLBACKS: TAB 2 (Targeted Dynamics)
    # =========================================================================

    @app.callback(
        Output("ego-heatmap", "figure"),
        Input("ego-tf-dropdown", "value"),
        Input("ego-corr-slider", "value"),
    )
    def update_1d_heatmap(tf_name, threshold):
        if not tf_name:
            return go.Figure()

        # 1. Fetch data
        ego_data = {
            t: data_loader.tf_correlation_dfs[t].loc[tf_name] for t in timepoints
        }
        ego_df = pd.DataFrame(ego_data)

        # 2. Remove the target TF itself so it doesn't just show a solid block of 1.0 correlation
        if tf_name in ego_df.index:
            ego_df = ego_df.drop(index=tf_name)

        # 3. Cluster
        ego_df = ego_df.loc[(ego_df.abs() > threshold).any(axis=1)]
        ego_df = order_dataframe_by_linkage(ego_df)

        # 4. Calculate dynamic height based on the full list of TFs
        plot_height = max(300, len(ego_df) * 20 + 100)

        # 5. Build Heatmap
        fig = go.Figure(
            go.Heatmap(
                z=ego_df.values,
                x=ego_df.columns,
                y=ego_df.index,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(ego_df.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Spearman", thickness=15),
                hovertemplate="<b>Partner TF:</b> %{y}<br><b>Time:</b> %{x}<br><b>Correlation:</b> %{z:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            height=plot_height,  # Apply dynamic height
            margin=dict(l=20, r=20, t=20, b=40),
            yaxis_autorange="reversed",
            xaxis=dict(
                title="Timepoints",
                showgrid=False,
                tickmode="array",
                tickvals=timepoints,
            ),
            yaxis=dict(title="Partner TFs", showgrid=False),
            plot_bgcolor="white",
        )
        return fig

    @app.callback(
        Output("global-expression-violin", "figure"),
        Output("violin-plot-title", "children"),
        Input("gene-picker", "value"),
        Input("downstream-tf-picker", "value"),
        Input("ego-tf-dropdown", "value"),
    )
    def update_global_violin(gene_target, tf_downstream, tf_ego):
        trigger_id = ctx.triggered_id

        # 1. Define the empty/waiting state
        empty_fig = go.Figure().update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text="Select a Target Gene or Regulator TF below<br>to view its baseline expression over time.",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color="gray", size=14),
                )
            ],
            plot_bgcolor="white",
            margin=dict(l=0, r=0, t=30, b=0),
        )

        # 2. Route the selection
        if not trigger_id:
            # Initial boot load: force the empty state
            return empty_fig, "Baseline Metacell Expression (Awaiting Selection)"

        elif trigger_id == "gene-picker":
            selected_entity = gene_target
            source_panel = "Downstream Target Gene"
        elif trigger_id == "downstream-tf-picker":
            selected_entity = tf_downstream
            source_panel = "Upstream Regulator TF"
        elif trigger_id == "ego-tf-dropdown":
            selected_entity = tf_ego
            source_panel = "Co-expressed TF"
        else:
            return empty_fig, "Baseline Metacell Expression (Awaiting Selection)"

        # Safety catch if a dropdown is cleared manually
        if not selected_entity:
            return empty_fig, "Baseline Metacell Expression (Awaiting Selection)"

        # 3. Build the actual Violin Plot
        fig = go.Figure()

        for t in timepoints:
            df_exp = data_loader.get_metacell_expression(t, [selected_entity])

            if not df_exp.empty and selected_entity in df_exp.columns:
                vals = df_exp[selected_entity].values

                fig.add_trace(
                    go.Violin(
                        x=[t] * len(vals),
                        y=vals,
                        name=t,
                        box_visible=True,
                        meanline_visible=True,
                        line_color="#5c5c5c",
                        fillcolor="rgba(92, 92, 92, 0.3)",
                        showlegend=False,
                        hovertemplate=f"<b>{t}</b><br>Expression: %{{y:.2f}}<extra></extra>",
                    )
                )

        fig.update_layout(
            margin=dict(l=30, r=20, t=10, b=30),
            plot_bgcolor="white",
            yaxis=dict(
                title="Expression",
                showgrid=True,          # Enable horizontal lines
                gridcolor="#e5e5e5",    # Soft light-grey color 
                zeroline=False          # Hides the thick 0 line if expression is log-normalized
            ),
            xaxis=dict(showgrid=False, tickmode="array", tickvals=timepoints),
        )

        title_text = f"Baseline Metacell Expression: {selected_entity} ({source_panel})"

        return fig, title_text

    @app.callback(
        Output("split-streamgraph", "figure"),
        Input("gene-picker", "value"),
        Input("ridge-threshold-slider", "value"),
    )
    def update_temporal_dot_plot(gene_name, threshold):
        if not gene_name:
            return go.Figure()

        df_coef = data_loader.get_in_edges_dataframe(gene_name, "coefficient")
        if df_coef.empty:
            return go.Figure()

        if "avg" in df_coef.columns:
            df_coef = df_coef.drop(columns=["avg"])

        # 1. FIX THE X-AXIS ORDER: explicitly subset columns in chronological order
        ordered_timepoints = [t for t in timepoints if t in df_coef.columns]
        df_coef = df_coef[ordered_timepoints]

        # 2. FILTER NOISE: Keep TFs with at least one significant timepoint
        # Apply dynamic noise threshold from the UI
        df_coef = df_coef.loc[(df_coef.abs() > threshold).any(axis=1)]

        if df_coef.empty:
            return go.Figure().update_layout(
                annotations=[
                    dict(
                        text="No significant upstream regulators found above the current noise threshold.",
                        showarrow=False,
                        font=dict(color="gray"),
                    )
                ],
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor="white",
            )

        # Cluster the dynamically filtered data on the fly!
        df_coef = order_dataframe_by_linkage(df_coef)

        plot_height = max(300, len(df_coef) * 20 + 100)

        # 3. CREATE THE HEATMAP
        fig = go.Figure(
            go.Heatmap(
                z=df_coef.values,
                x=df_coef.columns,
                y=df_coef.index,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-global_max_coef,
                zmax=global_max_coef,
                colorbar=dict(title="Ridge<br>Coefficient", thickness=15),
                # Optional but highly recommended: overlay the exact numbers on the cells
                text=np.round(df_coef.values, 3),
                texttemplate="%{text}",
                hovertemplate="<b>TF:</b> %{y}<br><b>Time:</b> %{x}<br><b>Coefficient:</b> %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            height=plot_height,
            margin=dict(l=20, r=20, t=40, b=40),
            xaxis=dict(
                title="Timepoints",
                showgrid=False,
                tickmode="array",
                tickvals=ordered_timepoints,
            ),
            yaxis=dict(title="Upstream TFs", showgrid=False, autorange="reversed"),
            plot_bgcolor="white",
        )

        return fig

    @app.callback(
        Output("downstream-heatmap", "figure"),
        Input("downstream-tf-picker", "value"),
        Input("downstream-corr-slider", "value"),
    )
    def update_downstream_heatmap(tf_name, threshold):
        if not tf_name:
            return go.Figure()

        # 1. Fetch the data using your native method
        try:
            df_targets = data_loader.get_out_edges_dataframe(
                tf_name, "correlation", threshold=0.1
            )
        except Exception:
            df_targets = pd.DataFrame()  # Safety catch if TF isn't in the network

        # Empty state fallback
        if df_targets.empty:
            return go.Figure().update_layout(
                annotations=[
                    dict(
                        text=f"No downstream targets found for {tf_name} above threshold.",
                        showarrow=False,
                        font=dict(color="gray"),
                    )
                ],
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor="white",
            )

        # 2. Clean up for plotting (Remove the 'avg' column and enforce chronology)
        if "avg" in df_targets.columns:
            df_targets = df_targets.drop(columns=["avg"])

        ordered_timepoints = [t for t in timepoints if t in df_targets.columns]
        df_targets = df_targets[ordered_timepoints]
        df_targets = df_targets.loc[(df_targets.abs() > threshold).any(axis=1)]

        df_targets = order_dataframe_by_linkage(df_targets)

        # 3. Calculate dynamic height (20px per gene row + 100px for margins/labels)
        plot_height = max(300, len(df_targets) * 20 + 100)

        # 4. Build the Heatmap
        fig = go.Figure(
            go.Heatmap(
                z=df_targets.values,
                x=df_targets.columns,
                y=df_targets.index,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,  # Spearman natively scales from -1 to 1
                zmax=1,
                colorbar=dict(title="Spearman<br>Correlation", thickness=15),
                text=np.round(df_targets.values, 2),
                texttemplate="%{text}",
                hovertemplate="<b>Target Gene:</b> %{y}<br><b>Time:</b> %{x}<br><b>Correlation:</b> %{z:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            height=plot_height,
            margin=dict(l=20, r=20, t=40, b=40),
            xaxis=dict(
                title="Timepoints",
                showgrid=False,
                tickmode="array",
                tickvals=ordered_timepoints,
            ),
            yaxis=dict(title="Target Genes", showgrid=False, autorange="reversed"),
            plot_bgcolor="white",
        )

        return fig

    app.run(debug=True)
