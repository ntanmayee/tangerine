import networkx as nx
import pandas as pd
from os.path import join
import numpy as np


class DataLoader(object):
    def __init__(self, timepoints, base_path, ) -> None:
        self.timepoints = timepoints
        self.base_path = base_path
        # load data
        self.networks = self.load_networks()
        self.tf_correlation_dfs = self.load_tf_correlations()
        self.tf_list = self.get_tf_list()
        self.tf_louvain = self.load_tf_louvain()

    def load_tf_louvain(self):
        return pd.read_csv(join(self.base_path, 'louvain_tf.csv'), index_col=0)

    def get_tf_list(self):
        time = self.timepoints[0]
        return list(self.tf_correlation_dfs[time].index)

    def load_tf_correlations(self):
        corr_dfs = {}
        for time in self.timepoints:
            corr_dfs[time] = pd.read_csv(join(self.base_path, f'tf_tf_{time}.csv'), index_col=0)
        return corr_dfs
    
    def get_parcat_dimensions(self):
        self.dimensions = [dict(values=self.tf_louvain[label], label=label) for label in self.timepoints]
        return self.dimensions

    def get_tf_corr_df(self, gene, tf_list):
        # make dataframe for this tf
        avg_corr = pd.DataFrame(index=tf_list)
        for time in self.timepoints:
            avg_corr[time] = self.tf_correlation_dfs[time].loc[gene, tf_list]       
        return avg_corr
    
    def get_tf_corr_basic(self, time):
        return self.tf_correlation_dfs[time], list(self.tf_correlation_dfs[time].index)

    def load_networks(self):
        networks = {}
        for time in self.timepoints:
            networks[time] = nx.read_gml(join(self.base_path, f'network_{time}.gml.gz'))
        return networks

    def get_out_edges_dataframe(self, tf_name, attr_name, threshold=0.1):
        index = [v for _,v in self.networks['0h'].out_edges(tf_name)]
        data_df = pd.DataFrame(index=index)

        for time in self.timepoints:
            values = []
            for gene in index:
                values.append(self.networks[time].get_edge_data(tf_name, gene)[attr_name])
            data_df[time] = values
        data_df['avg'] = data_df.mean(axis=1)
        
        return data_df[np.abs(data_df.avg) > threshold]

    def get_in_edges_dataframe(self, gene_name, attr_name, threshold=0.05):
        index = [u for u,_ in self.networks['0h'].in_edges(gene_name)]
        data_df = pd.DataFrame(index=index)
        
        for time in self.timepoints:
            values = []
            for tf_name in index:
                values.append(self.networks[time].get_edge_data(tf_name, gene_name)[attr_name])
            data_df[time] = values 
        data_df['avg'] = data_df.mean(axis=1)
        
        return data_df[np.abs(data_df.avg) > threshold]

    