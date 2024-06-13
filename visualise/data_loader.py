import networkx as nx
import pandas as pd
from os.path import join
import matplotlib as mpl
from matplotlib import colors
import numpy as np


class DataLoader(object):
    def __init__(self, timepoints, base_path, ) -> None:
        self.timepoints = timepoints
        self.base_path = base_path
        # load data
        self.networks = self.load_networks()
        self.tf_correlation_dfs = self.load_tf_correlations()
        self.tf_list = list(self.tf_correlation_dfs[0].index)

    def load_tf_correlations(self):
        corr_dfs = {}
        for time in self.timepoints:
            corr_dfs[time] = pd.read_csv(join(self.base_path, f'tf_tf_{time}.csv'), index_col=0)
        return corr_dfs

    def get_tf_corr_df(self, gene, tf_list):
        # make dataframe for this tf
        avg_corr = pd.DataFrame(index=tf_list)
        for time in self.timepoints:
            avg_corr[time] = self.tf_correlation_dfs[time].loc[gene, tf_list]       
        return avg_corr

    def load_networks(self):
        networks = {}
        for time in self.timepoints:
            networks[time] = nx.read_gml(join(self.base_path, f'network_{time}.gml.gz'))
        return networks

    def get_out_edges_dataframe(self, tf_name, attr_name):
        index = [v for _,v in self.networks['0h'].out_edges(tf_name)]
        data_df = pd.DataFrame(index=index)

        for time in self.timepoints:
            values = []
            for gene in index:
                values.append(self.networks[time].get_edge_data(tf_name, gene)[attr_name])
            data_df[time] = values
        
        return data_df

    def get_in_edges_dataframe(self, gene_name, attr_name):
        index = [u for u,_ in self.networks['0h'].in_edges(gene_name)]
        data_df = pd.DataFrame(index=index)
        
        for time in self.timepoints:
            values = []
            for tf_name in index:
                values.append(self.networks[time].get_edge_data(tf_name, gene_name)[attr_name])
            data_df[time] = values
        
        return data_df