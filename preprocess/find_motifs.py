from gimmemotifs.scanner import Scanner
from gimmemotifs.motif import read_motifs
from gimmemotifs.config import MotifConfig
import scanpy as sc
from utils import peak2fasta, scan_dna_for_motifs
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import networkx as nx
from biothings_client import get_client
from os.path import join
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import os
import pathlib
from logger import logger


# copied from previous version of gimme motifs repository
# somehow this has been removed in the lastest update

def default_motifs():
    """Return list of Motif instances from default motif database."""
    config = MotifConfig()
    d = config.get_motif_dir()
    m = config.get_default_params()["motif_db"]

    if not d or not m:
        raise ValueError("default motif database not configured")

    fname = os.path.join(d, m)
    with open(fname) as f:
        motifs = read_motifs(f)

    return motifs


class ScannerMotifs(object):
    def __init__(self, genome, n_cpus=1, fpr=0.02, scan_width=5000) -> None:
        self.genome = genome

        self.scanner_object = Scanner(ncpus=n_cpus)
        self.motifs = default_motifs()

        self.scanner_object.set_background(genome=genome)
        self.scanner_object.set_motifs(self.motifs)
        self.scanner_object.set_threshold(fpr=fpr)

        self.scan_width = scan_width
        self.all_tfs = self.get_all_motifs()

    def get_all_motifs(self):
        all_tfs = []

        for motif in self.motifs:
            all_tfs.extend(motif.factors['direct'])
            all_tfs.extend(motif.factors['indirect\nor predicted'])

        all_tfs = [a.capitalize() for a in all_tfs]
        all_tfs = list(set(all_tfs))
        return all_tfs

    def scan_motifs_single(self, chr_name, start, strand):
        peakstr = f'{chr_name}_{start - (self.scan_width * strand)}_{start}'
        target_sequences = peak2fasta(peak_ids=[peakstr], ref_genome=self.genome)
        scanned_df = scan_dna_for_motifs(scanner_object=self.scanner_object, motifs_object=self.motifs, 
                                      sequence_object=target_sequences, divide=100000)
        
        return scanned_df
    
    def get_filtered_tfs(self, scanned_df, score=None, include_indirect=False):
        if score:
            assert score > 0, "Invalid score, cannot filter binding sites"
            scanned_df = scanned_df[scanned_df.score > score]
        
        temp_tf_list = list(scanned_df.factors_direct.unique())
        if include_indirect:
            temp_tf_list += list(scanned_df.factors_indirect.unique())

        tf_list = []

        for element in temp_tf_list:
            temp = element.split(',')
            for e in temp:
                if len(e.strip()) > 0:
                    tf_list.append(e.strip().capitalize())

        tf_list = list(set(tf_list))
        return tf_list
    
    def run_single_scan(self, chr_name, start, strand, score=None, include_indirect=False):
        scanned_df = self.scan_motifs_single(chr_name, start, strand)
        tf_list = self.get_filtered_tfs(scanned_df, score, include_indirect)
        return tf_list


class Network(object):
    def __init__(self, path_to_adata, timepoints, genome, time_var, n_pcs=35, n_neighbors=100, scan_width=10000) -> None:
        if genome in ['mm10', 'hg19']:
            self.genome = genome
            logger.info(f'{self.genome} selected')
        else:
            raise NotImplementedError('Unsupported organism. Please raise an issue if you think this is a mistake.')
        
        self.time_var = time_var
        self.path_to_adata = path_to_adata
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        self.timepoints = timepoints
        self.networks = {t:nx.DiGraph() for t in self.timepoints}
        
        self.preprocess_adata()
        self.scanner = ScannerMotifs(self.genome, scan_width=scan_width)

    def __str__(self) -> str:
        print_str = f'Tangerine Network object with timepoints {str(self.timepoints)}\n\n{str(self.adata)}'
        return print_str
    
    def __repr__(self) -> str:
        return self.__str__()

    def preprocess_adata(self):
        if self.path_to_adata.endswith('.h5ad'):
            self.adata = sc.read_h5ad(self.path_to_adata)
            logger.info('Succesfully read in adata')
        else:
            raise NotImplementedError('Cannot read single cell data file.')
        
        # check if pca and umap already exists, otherwise preprocess
        if 'pca' not in self.adata.uns.keys() or 'umap' not in self.adata.uns.keys():
            # count normalization to 10,000 reads per cell 
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            self.adata.raw = self.adata.copy()

            sc.pp.log1p(self.adata)
            sc.tl.pca(self.adata, svd_solver='arpack')
            sc.pp.neighbors(self.adata, n_pcs=self.n_pcs, metric='euclidean', n_neighbors=self.n_neighbors)
            sc.tl.umap(self.adata)

    def filter_selected_tfs(self, tf_list, dropout_threshold=50):
        filtered_tfs = []

        for tf in tf_list:
            try:
                dropout = self.adata.var.loc[tf]['pct_dropout_by_counts'] 
                if dropout <= dropout_threshold:
                    filtered_tfs.append(tf)
            except KeyError:
                pass
                
            try:
                dropout = self.adata.var.loc[tf.upper()]['pct_dropout_by_counts'] 
                if dropout <= dropout_threshold:
                    filtered_tfs.append(tf)
            except KeyError:
                pass
        
        filtered_tfs = list(set(filtered_tfs))
        logger.info(f'Total number of TFs: {len(filtered_tfs)}')
        return filtered_tfs

    def run_single_gene(self, gene_name, chr_name, start, strand, score=None, include_indirect=False, dropout_threshold=50):
        assert gene_name in self.adata.var.index, f'Gene {gene_name} not found in self.adata'
        assert chr_name not in self.scanner.all_tfs, 'Target gene cannot be a transcription factor'

        try:
            tf_list = self.scanner.run_single_scan(chr_name, start, strand, score, include_indirect)
        except:
            logger.info(f'Some error for gene {gene_name}, {chr_name}')
            return pd.DataFrame(), pd.DataFrame()
        tf_list = self.filter_selected_tfs(tf_list, dropout_threshold)

        if len(tf_list) < 1:
            return None, None

        correlation_df = pd.DataFrame(index=tf_list)    
        coefficients_df = pd.DataFrame(index=tf_list)

        for time in self.timepoints:
            data_df = sc.get.obs_df(self.adata, [gene_name] + tf_list, use_raw=True)
            data_df = data_df[data_df.index.map(lambda x: time in x)]
            
            # compute correlations
            correlation_df[time] = data_df.corr(method='spearman')[gene_name]

            # compute coefficients
            X_train, X_test, y_train, y_test = train_test_split(data_df[tf_list], data_df[gene_name], test_size=0.1, random_state=29)

            model = Ridge()
            model.fit(X_train, y_train)
            
            coefficients_df[time] = list(model.coef_)
            model_score = model.score(X_test, y_test)

            # update networkx object with correlation and coefficients
            network = self.networks[time]
            for tf in tf_list:
                network.add_edge(tf, gene_name, correlation=correlation_df.loc[tf][time], coefficient=coefficients_df.loc[tf][time])
        
            # set model score as node attribute
            nx.set_node_attributes(network, {gene_name: {'score': model_score}})

        return correlation_df, coefficients_df
    
    def check_path_exists(self, path_name):
        path = pathlib.Path(path_name)
        path.mkdir(parents=True, exist_ok=True)

    def run_all_genes(self, save_path, dropout_threshold=50):
        logger.info('Runing for all genes...')
        self.check_path_exists(save_path)

        filtered_genes = list(self.adata.var[self.adata.var['pct_dropout_by_counts'] < dropout_threshold].index)
        filtered_genes = list(filter(lambda x: x not in self.scanner.all_tfs, filtered_genes))

        mg = get_client('gene')
        result_df = mg.querymany(filtered_genes, scopes=['symbol'], fields=['genomic_pos.chr', 'genomic_pos.start', 'genomic_pos.strand'], as_dataframe=True, species="mouse")
        result_df.reset_index(inplace=True)
        result_df.rename({'query':'symbol'}, axis=1, inplace=True)
        result_df.dropna(subset=['genomic_pos.chr', 'genomic_pos.start', 'genomic_pos.strand'], inplace=True)
        result_df.drop_duplicates(subset=['symbol'], inplace=True)

        print(f'Total genes selected: {len(result_df)}')

        def run_gene_pandas(row):
            start = int(float(row['genomic_pos.start']))
            strand = int(row['genomic_pos.strand'])
            return self.run_single_gene(row.symbol, f'chr{row["genomic_pos.chr"]}', start, strand)
        
        result_df.apply(run_gene_pandas, axis=1)
        logger.info('Complete!')

        Path(save_path).mkdir(parents=True, exist_ok=True)
        for time in self.timepoints:
            nx.write_gml(self.networks[time], join(save_path, f'network_{time}.gml.gz'))
        logger.info('Saved result graphs into file.')

    def run_tf_correlation(self, save_path, dropout_threshold=50):
        self.check_path_exists(save_path)

        def filter_selected_tfs(tf_list, dropout_threshold=50):
            filtered_tfs = []

            for tf in tf_list:
                try:
                    dropout = self.adata.var.loc[tf]['pct_dropout_by_counts'] 
                    if dropout <= dropout_threshold:
                        filtered_tfs.append(tf)
                except KeyError:
                    pass

                try:
                    dropout = self.adata.var.loc[tf.upper()]['pct_dropout_by_counts'] 
                    if dropout <= dropout_threshold:
                        filtered_tfs.append(tf)
                except KeyError:
                    pass
            
            return filtered_tfs

        filt_tfs = filter_selected_tfs(self.scanner.all_tfs, dropout_threshold)
        
        # compute correlation
        logger.info('Computing correlation among TFs...')
        corr_dfs = {}
        for time in self.timepoints:
            data_df = sc.get.obs_df(self.adata, filt_tfs + [self.time_var], use_raw=False)
            data_df = data_df[data_df[self.time_var] == time].drop(self.time_var, axis=1)
            corr_dfs[time] = data_df.corr('spearman')

            # write to file
            corr_dfs[time].to_csv(join(save_path, f'tf_tf_{time}.csv'))

        logger.info('Clustering TFs...')
        # cluster using heirarchical clustering
        cluster_df = pd.DataFrame(index=corr_dfs[self.timepoints[0]].index, columns=self.timepoints)

        for time in self.timepoints:
            X = corr_dfs[time].values
            Z = linkage(X, method='ward')

            # optimise using elbow-like metric for distances
            distances = Z[:, 2]
            diffs = np.diff(distances)

            max_gap_idx = np.argmax(diffs)
            num_clusters = len(distances) - max_gap_idx
            
            cluster_labels = fcluster(Z, t=num_clusters, criterion='maxclust')
            cluster_df.loc[corr_dfs[time].index, time] = cluster_labels

        for time in self.timepoints:
            cluster_df[time] = cluster_df[time].apply(lambda x: f'{time}-{x}')
            logger.debug(cluster_df[time].value_counts())

        # write to file
        cluster_df.to_csv(join(save_path, 'louvain_tf.csv'))
        logger.info('Saved clustering results to file.')
