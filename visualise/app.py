import numpy as np
import json
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from data_loader import DataLoader
from matplotlib import colors
import matplotlib as mpl
import itertools as it


def make_app_layout(app, tf_list_picker, gene_list, timepoints):
    # App layout
    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

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
                        html.P('Spearman correlation with all other genes. Select a timepoint from the dropdown menu.'),
                        dcc.Dropdown(timepoints, '0h', id='time-picker'),
                        dcc.Graph(figure={}, id='tf-heatmap'),
                        dcc.Store(id='tick-values')
                    ]),
                    dbc.Col([
                        html.H5('Radial Ego Network', style={"margin-top": "20px"}),
                        html.P('Inspect correlation of one TF with the rest. Select a TF from the dropdown menu.'),
                        dcc.Dropdown(tf_list_picker, 'Pou5f1', id='tf-picker-radial'),
                        dcc.Graph(figure={}, id='tf-tf', config={'displayModeBar': False, 'scrollZoom': False})
                    ]),
                ]),
                dbc.Row([
                    html.H5('Alluvial plot'),
                    dcc.Graph(figure={}, id='parcat'),
                    dbc.Table(id='output')

                ])
            ]),
            dcc.Tab(label='TF-Gene', children=[
                dbc.Row([
                    dbc.Col([
                        html.H5('TF view'),
                        html.P('Select a TF from the dropdown menu.'),
                        dcc.Dropdown(tf_list_picker, 'Pou5f1', id='tf-picker'),
                        dcc.Graph(figure={}, id='tf-correlation'),
                        html.Pre(id='selected-genes', style=styles['pre'])
                    ]),
                    dbc.Col([
                        html.H5('Gene view'),
                        html.P('Select a gene from the dropdown menu.'),
                        dcc.Dropdown(gene_list, 'L1td1', id='gene-picker'),
                        dcc.Graph(figure={}, id='gene-correlation'),
                        html.Pre(id='selected-tfs', style=styles['pre'])
                    ])
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

    # tick_names = list(data_loader.tf_correlation_dfs['0h'].index)

    # parcat init
    color_parcat = np.zeros(len(data_loader.tf_louvain), dtype='uint8')
    colorscale_parcat = [[0, 'gray'], [1, 'firebrick']]

    tf_list_picker = []
    for node in data_loader.networks['0h']:
        if len(data_loader.networks['0h'].out_edges(node)) > 0:
            tf_list_picker.append(node)
                
    gene_list = []
    for node in data_loader.networks['0h']:
        if len(data_loader.networks['0h'].in_edges(node)) > 0:
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

        # Helper to compute which category centers fall inside selection box
        def get_selected_labels(axis_labels, selected_range):
            selected_labels = []
            for i, label in enumerate(axis_labels):
                center = i  # Category i is centered at i
                left = center - 0.5
                right = center + 0.5
                # Check if cell overlaps with selected range
                if right >= selected_range[0] and left <= selected_range[1]:
                    selected_labels.append(label)
            return selected_labels

        selected_x_labels = get_selected_labels(tick_names, x_range)
        selected_y_labels = get_selected_labels(tick_names, y_range)

        selected_coords = sorted(list(set(list(selected_x_labels + selected_y_labels))))

        new_color = np.zeros(len(data_loader.tf_louvain), dtype='uint8')
        new_indices = [data_loader.tf_louvain.index.get_loc(tf) for tf in selected_coords]
        print(f'new_indices: {new_indices}')
        new_color[new_indices] = 1
        print(new_color, np.sum(new_color)) 

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

        # Helper to compute which category centers fall inside selection box
        def get_selected_labels(axis_labels, selected_range):
            selected_labels = []
            for i, label in enumerate(axis_labels):
                center = i  # Category i is centered at i
                left = center - 0.5
                right = center + 0.5
                # Check if cell overlaps with selected range
                if right >= selected_range[0] and left <= selected_range[1]:
                    selected_labels.append(label)
            return selected_labels

        selected_x_labels = get_selected_labels(tick_names, x_range)
        selected_y_labels = get_selected_labels(tick_names, y_range)

        selected_coords = sorted(list(set(list(selected_x_labels + selected_y_labels))))

        print("Selected Heatmap Coordinates (x, y):", selected_coords)
        temp_df = data_loader.tf_louvain.loc[selected_coords]
        temp_df['gene'] = temp_df.index
        # return f"{len(selected_coords)} TFs selected: {', '.join(selected_coords)}"
        return dbc.Table.from_dataframe(temp_df)


    @callback(
            Output(component_id='tf-heatmap', component_property='figure'),
            Output(component_id='tick-values', component_property='data'),
            Input(component_id='time-picker', component_property='value')
    )
    def update_tf_heatmap(time):
        corr_df, tick_names = data_loader.get_tf_corr_basic(time)
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
                # hoverongaps=False
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
        corr_df = data_loader.get_tf_corr_df(gene, data_loader.tf_list)
        print(corr_df.sort_values('0h', key=abs, ascending=False).head())
        tf_list = list(corr_df.sort_values('0h', ascending=False).index)
        print(tf_list)
        tf_list.remove(gene)

        # define coordinates and colours to plot
        offset = 2
        radii = list(range(offset, len(timepoints) +1 + offset))
        radii = [r*3 for r in radii]

        r_coords = list(it.pairwise(radii)) * len(tf_list)
        angle_coords = []
        for angle in np.linspace(0, 360, num=len(tf_list)+1)[:-1]:
            angle_coords.extend([angle] * (len(timepoints)))

        # colours 
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
                        mode='lines', line={'color': color},
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

        max_radius = np.max(r_coords) + 2.5

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
        df_coef = data_loader.get_out_edges_dataframe(tf_name, 'coefficient')
        values_range_coef = (np.min(df_coef), np.max(df_coef))
        
        fig = go.Figure(data=
                        go.Parcoords(
                            line_color = '#3182bd',
                            dimensions = list([
                                dict(range = values_range_coef,
                                    label = '0h coefficient',
                                    values = df_coef['0h']
                                    ),
                                dict(range = values_range_coef,
                                    label = '6h coefficient',
                                    values = df_coef['6h']
                                    ),
                                dict(range = values_range_coef,
                                    label = '18h coefficient',
                                    values = df_coef['18h']
                                    ),
                                dict(range = values_range_coef,
                                    label = '54h coefficient',
                                    values = df_coef['54h']
                                    ),
                            ])
                        )
                    )

        return fig

    @callback(
        Output(component_id='selected-genes', component_property='children'),
        State(component_id='tf-correlation', component_property='figure'),
        Input(component_id='tf-correlation', component_property='restyleData'),
        Input(component_id='tf-picker', component_property='value')
    )
    def update_gene_list(state, _, tf_name):
        # state = json.loads(state)
        df_corr = data_loader.get_out_edges_dataframe(tf_name, 'correlation')
        df_coef = data_loader.get_out_edges_dataframe(tf_name, 'coefficient')
        merged = df_coef.join(df_corr, rsuffix='_correlation', lsuffix='_coefficient')
        
        changed_cols = []
        
        try:
            for i, dim in enumerate(state['data'][0]['dimensions']):
                try:
                    mini, maxi = dim['constraintrange']
                    merged = merged[ (merged[merged.columns[i]] <= maxi) & (merged[merged.columns[i]] >= mini) ]
                    changed_cols.append(merged.columns[i])
                except KeyError:
                    continue
        except KeyError:
            try:
                return f'{tf_name}, {dim}, {changed_cols}, {json.dumps(state)}'
            except:
                return f'{tf_name}, {json.dumps(state)}'
            # return 'Some error!'
        
        genes = list(merged.index)
        
        if len(genes) < 10:
            base_url = 'https://www.genecards.org/cgi-bin/carddisp.pl?gene='
            return [dbc.NavLink(gene, href=base_url+gene, external_link=True) for gene in genes]

        return f'Too many genes to display. Try to reduce your selection.'

    @callback(
        Output(component_id='gene-correlation', component_property='figure'),
        Input(component_id='gene-picker', component_property='value')
    )
    def update_gene_graph(gene_name):
        df_corr = data_loader.get_in_edges_dataframe(gene_name, 'correlation')
        df_coef = data_loader.get_in_edges_dataframe(gene_name, 'coefficient')
        values_range_corr = (np.min(df_corr), np.max(df_corr))
        values_range_coef = (np.min(df_coef), np.max(df_coef))
        
        fig = go.Figure(data=
                        go.Parcoords(
                            line_color = '#3182bd',
                            dimensions = list([
                                # dict(range = values_range_coef,
                                #     label = '0h coefficient',
                                #     values = df_coef['0h']
                                #     ),
                                # dict(range = values_range_coef,
                                #     label = '6h coefficient',
                                #     values = df_coef['6h']
                                #     ),
                                # dict(range = values_range_coef,
                                #     label = '18h coefficient',
                                #     values = df_coef['18h']
                                #     ),
                                # dict(range = values_range_coef,
                                #     label = '54h coefficient',
                                #     values = df_coef['54h']
                                #     ),
                                dict(range = values_range_corr,
                                    label = '0h correlation',
                                    values = df_corr['0h']
                                    ),
                                dict(range = values_range_corr,
                                    label = '6h correlation',
                                    values = df_corr['6h']
                                    ),
                                dict(range = values_range_corr,
                                    label = '18h correlation',
                                    values = df_corr['18h']
                                    ),
                                dict(range = values_range_corr,
                                    label = '54h correlation',
                                    values = df_corr['54h']
                                    ),
                            ])
                        )
                    )

        return fig

    @callback(
        Output(component_id='selected-tfs', component_property='children'),
        State(component_id='gene-correlation', component_property='figure'),
        Input(component_id='gene-correlation', component_property='restyleData'),
        Input(component_id='gene-picker', component_property='value')
    )
    def update_gene_list(state, _, gene_name):
        # state = json.loads(state)
        df_corr = data_loader.get_in_edges_dataframe(gene_name, 'correlation')
        df_coef = data_loader.get_in_edges_dataframe(gene_name, 'coefficient')
        merged = df_corr.join(df_coef, rsuffix='_correlation', lsuffix='_coefficient')
        
        changed_cols = []
        
        try:
            for i, dim in enumerate(state['data'][0]['dimensions']):
                try:
                    mini, maxi = dim['constraintrange']
                    merged = merged[ (merged[merged.columns[i]] <= maxi) & (merged[merged.columns[i]] >= mini) ]
                    changed_cols.append(merged.columns[i])
                except KeyError:
                    continue
        except KeyError:
            try:
                return f'{gene_name}, {dim}, {changed_cols}, {json.dumps(state)}'
            except:
                return f'{gene_name}, {json.dumps(state)}'
            # return 'Some error!'
        
        genes = list(merged.index)
        
        if len(genes) < 10:
            base_url = 'https://www.genecards.org/cgi-bin/carddisp.pl?gene='
            return [dbc.NavLink(gene, href=base_url+gene, external_link=True) for gene in genes]

        return f'Too many genes to display. Try to reduce your selection.'
    
    app.run(debug=True)
