from gimmemotifs.scanner import Scanner
from gimmemotifs.fasta import Fasta
from gimmemotifs.motif import default_motifs
import scanpy as sc
from .utils import peak2fasta, scan_dna_for_motifs
import pandas as pd
from sklearn.linear_model import Ridge
import networkx as nx

class ScannerMotifs(object):
    def __init__(self, genome='mm10', n_cpus=1, fpr=0.02, scan_width=5000) -> None:
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

    def scan_motifs_single(self, chr_name, start):
        peakstr = f'{chr_name}:{start - self.scan_widthh}-{start}'
        target_sequences = peak2fasta(peak_ids=[peakstr], ref_genome='mm10')
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
    
    def run_single_scan(self, chr_name, start, score=None, include_indirect=False):
        scanned_df = self.scan_motifs_single(chr_name, start)
        tf_list = self.get_filtered_tfs(scanned_df, score, include_indirect)
        return tf_list


class Network(object):
    def __init__(self, path_to_adata, timepoints, n_pcs=35, n_neighbors=100) -> None:
        self.path_to_adata = path_to_adata
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        self.timepoints = timepoints
        self.networks = {t:nx.DiGraph() for t in self.timepoints}
        
        self.preprocess_adata()
        self.scanner = ScannerMotifs()

    def preprocess_adata(self):
        if self.path_to_adata.endswith('.h5ad'):
            self.adata = sc.read_h5ad(self.path_to_adata)

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
        return filtered_tfs

    def run_single_gene(self, gene_name, chr_name, start, score=None, include_indirect=False, dropout_threshold=50):
        assert chr_name not in self.scanner.all_tfs, 'Target gene cannot be a transcription factor'

        tf_list = self.scanner.run_single_scan(chr_name, start, score, include_indirect)
        tf_list = self.filter_selected_tfs(tf_list, dropout_threshold)

        correlation_df = pd.DataFrame(index=tf_list)    
        coefficients_df = pd.DataFrame(index=tf_list)

        for time in self.timepoints:
            data_df = sc.get.obs_df(self.adata, [gene_name] + tf_list, use_raw=True)
            data_df = data_df[data_df.index.map(lambda x: time in x)]
            
            # compute correlations
            correlation_df[time] = data_df.corr(method='spearman')[gene_name]

            # compute coefficients
            model = Ridge()
            model.fit(data_df[tf_list], data_df[gene_name])
            coefficients_df[time] = list(model.coef_)

        # update networkx object with correlation and coefficients
        for time in self.timepoints:
            network = self.networks[time]
            for tf in tf_list:
                network.add_edge(tf, gene_name, correlation=correlation_df.loc[tf][time], coefficient=coefficients_df.loc[tf][time])

        return correlation_df, coefficients_df