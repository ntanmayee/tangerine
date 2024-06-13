import networkx as nx
import pandas as pd
import numpy as np
import json
from os.path import join

from dash import Dash, html, dcc, callback, Output, Input, State
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


def make_app_layout(app, tf_list, gene_list):
    # App layout
    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

    app.layout = dbc.Container([
        dbc.Row([
            html.H1('Visualising Gene Regulatory Networks'),
            html.P('This is an interactive visualisation of transcription factor -> gene association networks.'),
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4('Transcription Factor Associations'),
                html.P('Select a TF from the dropdown menu.'),
                dcc.Dropdown(tf_list, 'Pou5f1', id='tf-picker'),
                html.H5('Coefficients of linear mixed model', style={"margin-top": "20px"}),
                html.P('The figure shows all genes associated with this TF. All shown genes have the motif of the selected TF in its promotor.'),
                dcc.Graph(figure={}, id='tf-correlation'),
                html.Pre(id='selected-genes', style=styles['pre'])
            ]),
            dbc.Col([
                html.H4('Gene Associations'),
                html.P('Select a gene from the dropdown menu.'),
                dcc.Dropdown(gene_list, 'L1td1', id='gene-picker'),
                html.H5('Spearman correlation', style={"margin-top": "20px"}),
                html.P("The figure shows all TFs associated with this gene. All shown TFs have their motif in the gene's promotor."),
                dcc.Graph(figure={}, id='gene-correlation'),
                html.Pre(id='selected-tfs', style=styles['pre'])
            ])
        ])
    ])

def run_app(timepoints, base_path):
    data_loader = DataLoader(timepoints, base_path)


    tf_list = []
    for node in data_loader.networks['0h']:
        if len(data_loader.networks['0h'].out_edges(node)) > 0:
            tf_list.append(node)
                
    gene_list = []
    for node in data_loader.networks['0h']:
        if len(data_loader.networks['0h'].in_edges(node)) > 0:
            gene_list.append(node)

    # Initialize the app
    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    make_app_layout(app, tf_list, gene_list)

    @callback(
        Output(component_id='tf-correlation', component_property='figure'),
        Input(component_id='tf-picker', component_property='value')
    )
    def update_tf_graph(tf_name):
        df_coef = data_loader.get_out_edges_dataframe(tf_name, 'coefficient')
        values_range_coef = (np.min(df_coef), np.max(df_coef))
        
        fig = go.Figure(data=
                        go.Parcoords(
                            line_color = 'blue',
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
        
        if len(genes) < 50:
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
                            line_color = 'blue',
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
        
        if len(genes) < 20:
            base_url = 'https://www.genecards.org/cgi-bin/carddisp.pl?gene='
            return [dbc.NavLink(gene, href=base_url+gene, external_link=True) for gene in genes]

        return f'Too many genes to display. Try to reduce your selection.'
    
    app.run(debug=True)

if __name__ == '__main__':
    timepoints = ['0h', '6h', '18h', '54h']
    base_path = 'arid1a_networks'
    run_app(timepoints, base_path)
