import numpy as np
import json
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from tangerine.visualise.data_loader import DataLoader
from matplotlib import colors
import matplotlib as mpl
import itertools as it
from scipy.cluster.hierarchy import linkage, leaves_list  # NEW: imported for on-the-fly visual ordering

def make_app_layout(app, tf_list_picker, gene_list, timepoints):
    # App layout
    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

    t0 = timepoints[0]
    tf_chosen = tf_list_picker[0] if tf_list_picker else None
    gene_chosen = gene_list[0] if gene_list else None

    app.layout = dbc.Container([
        dbc.Row([
            html.H1('Tangerine: Visualising Dynamic Gene Regulation'),
            html.P('This is an interactive visualisation of multi-omic time course data.'),
        ]),

        dcc.Tabs([
            dcc.Tab(label='TF Co-regulation', children=[
                dbc.Row([
                    html.H3('Transcription Factor Co-regulation'),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H5('Heatmap', style={"margin-top": "20px"}),
                        html.P('Spearman correlation with all other TFs. Select a timepoint from the dropdown menu.'),
                        dcc.Dropdown(timepoints, t0, id='time-picker'),
                        
                        html.Div([
                            dbc.RadioItems(
                                id='heatmap-order',
                                options=[
                                    {'label': f'Order by Initial Timepoint ({t0})', 'value': 't0'},
                                    {'label': 'Order by Current Timepoint', 'value': 'current'}
                                ],
                                value='t0',
                                inline=True,
                                style={'margin-top': '15px', 'margin-bottom': '5px', 'font-size': '14px'}
                            )
                        ]),
                        
                        dcc.Graph(figure={}, id='tf-heatmap'),
                        dcc.Store(id='tick-values')
                    ]),
                    dbc.Col([
                        html.H5('Radial Ego Network', style={"margin-top": "20px"}),
                        html.P('Inspect correlation of one TF with the rest. Select a TF from the dropdown menu.'),
                        dcc.Dropdown(tf_list_picker, tf_chosen, id='tf-picker-radial'),
                        dcc.Graph(figure={}, id='tf-tf', config={'displayModeBar': False, 'scrollZoom': False})
                    ]),
                ]),
                dbc.Row([
                    html.H5('Alluvial plot'),
                    html.P('Select a region from the heatmap to check how cluster membership changes over time.'),
                    dcc.Graph(figure={}, id='parcat'),
                    dbc.Table(id='output')
                ])
            ]),
            dcc.Tab(label='TF-Gene', children=[
                dbc.Row([
                    html.H5('TF view'),
                    html.P('Select a TF from the dropdown menu.'),
                    dcc.Dropdown(tf_list_picker, tf_chosen, id='tf-picker'),
                    dcc.Graph(figure={}, id='tf-correlation'),
                    html.Pre(id='selected-genes', style=styles['pre'])
                ]),
                dbc.Row([
                    html.H5('Gene view'),
                    html.P('Select a gene from the dropdown menu.'),
                    dcc.Dropdown(gene_list, gene_chosen, id='gene-picker'),
                    dcc.Graph(figure={}, id='gene-correlation'),
                    html.Pre(id='selected-tfs', style=styles['pre'])
                ]),
            ]),     
        ]),
    ])

def run_app(timepoints, base_path):
    data_loader = DataLoader(timepoints, base_path)

    # define colourmap and divnorm
    vmin, vcenter, vmax = -0.2, 0.0, 0.3
    divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = mpl.cm.RdBu
    cmap_ = mpl.cm.get_cmap('RdBu_r', 256)
    normed_vals = np.linspace(vmin, vmax, 256)
    rgba_colors = [cmap_(divnorm(v)) for v in normed_vals]
    colorscale = [
        [i / 255, f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.3f})']
        for i, (r, g, b, a) in enumerate(rgba_colors)
    ]

    # parcat init
    color_parcat = np.zeros(len(data_loader.tf_louvain), dtype='uint8')
    colorscale_parcat = [[0, 'gray'], [1, 'firebrick']]

    t0 = timepoints[0]
    tf_list_picker = []
    for node in data_loader.networks[t0]:
        if len(data_loader.networks[t0].out_edges(node)) > 0:
            tf_list_picker.append(node)
                
    gene_list = []
    for node in data_loader.networks[t0]:
        if len(data_loader.networks[t0].in_edges(node)) > 0:
            gene_list.append(node)

    # Initialize the app
    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    make_app_layout(app, tf_list_picker, gene_list, timepoints)

    @app.callback(
            Output('parcat', 'figure'),
            Input('tf-heatmap', 'selectedData'),
            Input('tick-values', 'data')
    )
    def update_parcat(selectedData, tick_names):
        dimensions = data_loader.get_parcat_dimensions()
        if not selectedData or 'range' not in selectedData:
            fig = go.Figure(
                    go.Parcats(
                        dimensions=dimensions,
                        line={'colorscale': colorscale_parcat, 'cmin': 0, 'cmax': 1, 'color': color_parcat, 'shape': 'hspline'}
                    )
            )
            return fig

        x_range = selectedData['range']['x']
        y_range = selectedData['range']['y']

        def get_selected_labels(axis_labels, selected_range):
            selected_labels = []
            for i, label in enumerate(axis_labels):
                center = i  
                left = center - 0.5
                right = center + 0.5
                if right >= selected_range[0] and left <= selected_range[1]:
                    selected_labels.append(label)
            return selected_labels

        selected_x_labels = get_selected_labels(tick_names, x_range)
        selected_y_labels = get_selected_labels(tick_names, y_range)

        selected_coords = sorted(list(set(list(selected_x_labels + selected_y_labels))))

        new_color = np.zeros(len(data_loader.tf_louvain), dtype='uint8')
        new_indices = [data_loader.tf_louvain.index.get_loc(tf) for tf in selected_coords if tf in data_loader.tf_louvain.index]
        new_color[new_indices] = 1

        fig = go.Figure(
            go.Parcats(
                        dimensions=dimensions,
                        line={'colorscale': colorscale_parcat, 'cmin': 0, 'cmax': 1, 'color': new_color, 'shape': 'hspline'}
                    )
        )
        return fig

    @app.callback(
        Output('output', 'children'),
        Input('tf-heatmap', 'selectedData'),
        Input('tick-values', 'data')
    )
    def handle_selection(selectedData, tick_names):
        if not selectedData or 'range' not in selectedData:
            return "Use box-select to choose heatmap cells."

        x_range = selectedData['range']['x']
        y_range = selectedData['range']['y']

        def get_selected_labels(axis_labels, selected_range):
            selected_labels = []
            for i, label in enumerate(axis_labels):
                center = i  
                left = center - 0.5
                right = center + 0.5
                if right >= selected_range[0] and left <= selected_range[1]:
                    selected_labels.append(label)
            return selected_labels

        selected_x_labels = get_selected_labels(tick_names, x_range)
        selected_y_labels = get_selected_labels(tick_names, y_range)

        selected_coords = sorted(list(set(list(selected_x_labels + selected_y_labels))))
        
        # Filter strictly for existing indices to avoid KeyErrors
        valid_coords = [c for c in selected_coords if c in data_loader.tf_louvain.index]
        temp_df = data_loader.tf_louvain.loc[valid_coords].copy()
        temp_df['gene'] = temp_df.index
        return dbc.Table.from_dataframe(temp_df)

    # --- UPDATED CALLBACK: Handles Heatmap Ordering ---
    @callback(
            Output(component_id='tf-heatmap', component_property='figure'),
            Output(component_id='tick-values', component_property='data'),
            Input(component_id='time-picker', component_property='value'),
            Input(component_id='heatmap-order', component_property='value') # NEW INPUT
    )
    def update_tf_heatmap(time, order_type):
        corr_df, tick_names = data_loader.get_tf_corr_basic(time)
        
        # If user wants current timepoint ordering, compute visual structure on the fly
        if order_type == 'current' and time != timepoints[0]:
            try:
                Z = linkage(corr_df.values, method='ward')
                optimal_order_indices = leaves_list(Z)
                # Reorder the dataframe and ticks
                corr_df = corr_df.iloc[optimal_order_indices, optimal_order_indices]
                tick_names = list(corr_df.columns)
            except Exception as e:
                print(f"Clustering failed for visual ordering: {e}")
                # Fallback to default t0 ordering if it fails
                pass

        fig = go.Figure(
            go.Heatmap(
                z=corr_df,
                x=tick_names,
                y=tick_names,
                colorscale=colorscale,
                zmin=-0.2,
                zmax=0.3,
                colorbar=dict(title="Value"),
                hoverinfo='skip'
            )
        )
        fig.update_layout(
            height=600,
            width=600,
            dragmode='select'
        )
        return fig, tick_names

    @callback(
        Output(component_id='tf-tf', component_property='figure'),
        Input(component_id='tf-picker-radial', component_property='value')
    )
    def update_tf_tf(gene):
        if not gene:
            return go.Figure()
            
        corr_df = data_loader.get_tf_corr_df(gene, data_loader.tf_list)
        tf_list = list(corr_df.sort_values(t0, ascending=False).index)
        if gene in tf_list:
            tf_list.remove(gene)

        offset = 2
        radii = list(range(offset, len(timepoints) +1 + offset))
        radii = [r*3 for r in radii]

        r_coords = list(it.pairwise(radii)) * len(tf_list)
        angle_coords = []
        for angle in np.linspace(0, 360, num=len(tf_list)+1)[:-1]:
            angle_coords.extend([angle] * (len(timepoints)))

        colours = corr_df.loc[tf_list].map(lambda x : colors.to_hex(cmap(divnorm(x)))).to_numpy().flatten()
        corr_values = corr_df.loc[tf_list].to_numpy().flatten()

        fig = go.Figure()

        for r, theta, color, corr in zip(r_coords, angle_coords, colours, corr_values):
            r1, r2 = r
            angle = np.deg2rad(theta)
            x_coords = [r1 * np.cos(angle), r2 * np.cos(angle)]
            y_coords = [r1 * np.sin(angle), r2 * np.sin(angle)]
            
            fig.add_trace(
                go.Scatter(x = x_coords, y = y_coords, 
                        mode='lines', line={'color': color, 'width': 4},
                        hoverinfo='none'
                        )
            )
            fig.add_trace(
                go.Scatter(x = [np.average(x_coords)], y = [np.average(y_coords)],
                        mode='markers', marker=dict(color='rgba(0, 0, 0, 0)'), 
                        text=f'{corr:.2f}',
                        hoverinfo='text'
                        )
            )

        max_radius = np.max(r_coords) + 2.5 if r_coords else 0

        for theta, tf in zip(np.linspace(0, 360, num=len(tf_list)+1)[:-1], tf_list):
            angle = np.deg2rad(theta)
            x, y = max_radius * np.cos(angle), max_radius * np.sin(angle)
            fig.add_annotation(x=x, y=y, text=tf, showarrow=False, textangle=-theta, xanchor="center", yanchor='middle', font={'size':8})
        
        fig.add_annotation(x=0, y=0, text=gene, showarrow=False, xanchor="center", yanchor='middle', font={'size':14})

        fig.update_layout(
                        autosize=False, width=500, height=500, 
                        margin=dict(l=5, r=5, t=15, b=5),
                        showlegend=False, template='plotly_white', 
                        xaxis_showticklabels=False, yaxis_showticklabels=False, 
                        xaxis_gridcolor="#ffffff", yaxis_gridcolor="#ffffff"
                        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)

        return fig

    @callback(
        Output(component_id='tf-correlation', component_property='figure'),
        Input(component_id='tf-picker', component_property='value')
    )
    def update_tf_graph(tf_name):
        if not tf_name:
            return go.Figure()
            
        df_coef = data_loader.get_out_edges_dataframe(tf_name, 'coefficient')
        if df_coef.empty:
            return go.Figure()
            
        values_range_coef = (np.min(df_coef.values), np.max(df_coef.values))

        dims = [
            dict(
                range = values_range_coef,
                label = f'{t} coefficient',
                values = df_coef[t]
            )
            for t in timepoints
        ]
        
        fig = go.Figure(data=
                        go.Parcoords(
                            line_color = '#3182bd',
                            dimensions = dims
                        )
                    )
        return fig

    @callback(
        Output(component_id='selected-genes', component_property='children'),
        State(component_id='tf-correlation', component_property='figure'),
        Input(component_id='tf-correlation', component_property='restyleData'),
        Input(component_id='tf-picker', component_property='value')
    )
    def update_tf_list(state, restyleData, tf_name):
        if not tf_name:
            return "No TF Selected"
            
        df_corr = data_loader.get_out_edges_dataframe(tf_name, 'correlation')
        df_coef = data_loader.get_out_edges_dataframe(tf_name, 'coefficient')
        merged = df_coef.join(df_corr, rsuffix='_correlation', lsuffix='_coefficient')
        
        if state and 'data' in state and state['data'] and 'dimensions' in state['data'][0]:
            for i, dim in enumerate(state['data'][0]['dimensions']):
                if 'constraintrange' in dim:
                    mini, maxi = dim['constraintrange']
                    # Some versions of plotly return a list of lists if multiple ranges are selected
                    if isinstance(mini, list):
                        pass # advanced handling can go here
                    else:
                        merged = merged[ (merged[merged.columns[i]] <= maxi) & (merged[merged.columns[i]] >= mini) ]
        
        genes = list(merged.index)
        
        if len(genes) < 10:
            base_url = 'https://www.genecards.org/cgi-bin/carddisp.pl?gene='
            return [dbc.NavLink(gene, href=base_url+gene, external_link=True) for gene in genes]

        return f'Selected {len(genes)} genes. Too many genes to display individually. Try to reduce your selection.'

    @callback(
        Output(component_id='gene-correlation', component_property='figure'),
        Input(component_id='gene-picker', component_property='value')
    )
    def update_gene_graph(gene_name):
        if not gene_name:
            return go.Figure()
            
        df_corr = data_loader.get_in_edges_dataframe(gene_name, 'correlation')
        df_coef = data_loader.get_in_edges_dataframe(gene_name, 'coefficient')
        
        if df_corr.empty or df_coef.empty:
            return go.Figure()
            
        values_range_corr = (np.min(df_corr.values), np.max(df_corr.values))

        dims = [
            dict(
                range = values_range_corr,
                label = f'{t} correlation',
                values = df_corr[t]
            )
            for t in timepoints
        ]
        
        fig = go.Figure(data=
                        go.Parcoords(
                            line_color = '#3182bd',
                            dimensions = dims
                        )
                    )
        return fig

    @callback(
        Output(component_id='selected-tfs', component_property='children'),
        State(component_id='gene-correlation', component_property='figure'),
        Input(component_id='gene-correlation', component_property='restyleData'),
        Input(component_id='gene-picker', component_property='value')
    )
    def update_gene_list(state, restyleData, gene_name):
        if not gene_name:
            return "No Gene Selected"
            
        df_corr = data_loader.get_in_edges_dataframe(gene_name, 'correlation')
        df_coef = data_loader.get_in_edges_dataframe(gene_name, 'coefficient')
        merged = df_corr.join(df_coef, rsuffix='_correlation', lsuffix='_coefficient')
        
        if state and 'data' in state and state['data'] and 'dimensions' in state['data'][0]:
            for i, dim in enumerate(state['data'][0]['dimensions']):
                if 'constraintrange' in dim:
                    mini, maxi = dim['constraintrange']
                    if not isinstance(mini, list):
                        merged = merged[ (merged[merged.columns[i]] <= maxi) & (merged[merged.columns[i]] >= mini) ]
        
        genes = list(merged.index)
        
        if len(genes) < 10:
            base_url = 'https://www.genecards.org/cgi-bin/carddisp.pl?gene='
            return [dbc.NavLink(gene, href=base_url+gene, external_link=True) for gene in genes]

        return f'Selected {len(genes)} TFs. Too many TFs to display. Try to reduce your selection.'
    
    app.run(debug=True)
