from gimmemotifs.scanner import Scanner
from gimmemotifs.fasta import Fasta
from gimmemotifs.motif import default_motifs
import scanpy as sc

class ScannerMotifs(object):
    def __init__(self, genome='mm10', n_cpus=1, fpr=0.02) -> None:
        self.scanner_object = Scanner(ncpus=n_cpus)
        self.scanner_object.set_background(genome=genome)
        motifs = default_motifs()
        self.scanner_object.set_motifs(motifs)
        self.scanner_object.set_threshold(fpr=fpr)


class Network(object):
    def __init__(self, path_to_adata, n_pcs=35, n_neighbors=100) -> None:
        self.path_to_adata = path_to_adata
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors

        self.adata = self.preprocess_adata()

    def preprocess_adata(self):
        if self.path_to_adata.endswith('.h5ad'):
            adata = sc.read_h5ad(self.path_to_adata)

        else:
            raise NotImplementedError('Cannot read single cell data file.')
        
        # check if pca and umap already exists, otherwise preprocess
        if 'pca' not in adata.uns.keys() or 'umap' not in adata.uns.keys():
            # count normalization to 10,000 reads per cell 
            sc.pp.normalize_total(adata, target_sum=1e4)
            adata.raw = adata.copy()

            sc.pp.log1p(adata)
            sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata, n_pcs=self.n_pcs, metric='euclidean', n_neighbors=self.n_neighbors)
            sc.tl.umap(adata)

        return adata
