import os
import pandas as pd
import scanpy as sc
import networkx as nx
from pathlib import Path

from tangerine.preprocess.metacells import KMeansMetacellGenerator, SEACellsMetacellGenerator
from tangerine.preprocess.motifs import ScannerMotifs
from tangerine.preprocess.inference import DynamicNetworkInference
from tangerine.preprocess.logger import logger

class TangerinePipeline:
    """
    The main orchestrator for the Tangerine preprocessing workflow.
    Handles data loading, metacell generation, motif scanning, and network inference.
    """
    def __init__(self, adata_path, timepoints, genome, time_var, save_path,
                 metacell_method='kmeans', n_pcs=35, n_neighbors=100, scan_width=10000):
        
        self.adata_path = adata_path
        self.timepoints = timepoints
        self.genome = genome
        self.time_var = time_var
        self.save_path = save_path
        self.scan_width = scan_width
        self.metacell_method = metacell_method
        
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        
        self.adata = None
        self.metacell_dict = {}
        
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

    def load_and_preprocess_data(self):
        """Loads the raw AnnData and runs basic scRNA-seq QC/PCA/UMAP."""
        logger.info(f"Loading data from {self.adata_path}...")
        if not self.adata_path.endswith('.h5ad'):
            raise ValueError('Currently only .h5ad files are supported.')
            
        self.adata = sc.read_h5ad(self.adata_path)
        sc.pp.calculate_qc_metrics(self.adata, inplace=True)

        # Standard Scanpy Preprocessing
        if 'pca' not in self.adata.uns.keys() or 'umap' not in self.adata.uns.keys():
            logger.info("Running standard preprocessing (Normalize, Log1p, PCA, UMAP)...")
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            self.adata.raw = self.adata.copy()
            sc.pp.log1p(self.adata)
            sc.tl.pca(self.adata, svd_solver='arpack')
            sc.pp.neighbors(self.adata, n_pcs=self.n_pcs, metric='euclidean', n_neighbors=self.n_neighbors)
            sc.tl.umap(self.adata)
        else:
            logger.info("PCA/UMAP already found in AnnData.")

    def generate_metacells(self):
        """Routes data to the selected Metacell Strategy."""
        if self.metacell_method.lower() == 'kmeans':
            generator = KMeansMetacellGenerator(self.adata, self.time_var, self.timepoints)
        elif self.metacell_method.lower() == 'seacells':
            generator = SEACellsMetacellGenerator(self.adata, self.time_var, self.timepoints)
        else:
            raise ValueError(f"Unknown metacell method: {self.metacell_method}")
            
        self.metacell_dict = generator.generate()

    def _save_metacells_to_parquet(self, tfs):
        """
        Filters the generated metacells to include only Transcription Factors,
        converts them to dense DataFrames, and saves them as fast-loading Parquet files.
        """
        logger.info("Saving TF-only metacell expression to Parquet for dashboard...")
        
        for time, m_data in self.metacell_dict.items():
            # Support both AnnData and raw Pandas DataFrames depending on what the generator yields
            if isinstance(m_data, sc.AnnData):
                df = m_data.to_df()
            else:
                df = pd.DataFrame(m_data)

            # Find which TFs actually exist in this specific expression matrix
            valid_tfs = [tf for tf in tfs if tf in df.columns]

            # Subset to just the valid TFs to save massive amounts of disk space
            df_tfs = df[valid_tfs]

            # Save to Parquet
            file_name = f"metacells_{time}.parquet"
            save_file = os.path.join(self.save_path, file_name)
            df_tfs.to_parquet(save_file)
            
            logger.info(f"Saved {file_name} with {len(valid_tfs)} TFs.")

    def _load_local_gene_annotations(self, bed_file_path, dropout_threshold=50):
        """
        Reads a standard BED file to get genomic coordinates.
        Expected columns: chrom, start, end, gene_symbol, score, strand
        """
        logger.info(f"Loading local gene annotations from {bed_file_path}...")
        
        # Load BED file (assuming 6 standard columns)
        df = pd.read_csv(bed_file_path, sep='\t', header=None, 
                         names=['chrom', 'start', 'end', 'symbol', 'score', 'strand'])
        
        # Filter for genes actually present in our data and passing dropout
        filtered_genes = list(self.adata.var[self.adata.var['pct_dropout_by_counts'] < dropout_threshold].index)
        
        # Keep only genes that exist in both the BED file and our filtered single-cell data
        df = df[df['symbol'].isin(filtered_genes)].drop_duplicates(subset=['symbol'])
        
        # Convert strand +/- to 1/-1
        df['strand_int'] = df['strand'].apply(lambda x: 1 if x == '+' else -1)
        
        logger.info(f"Found valid coordinates for {len(df)} expressed genes.")
        return df

    def run(self, bed_file_path, dropout_threshold=50):
        """The main execution loop for the pipeline."""
        
        # 1. Prepare Data
        self.load_and_preprocess_data()
        self.generate_metacells()
        
        # 2. Initialize Prior Scanner & Statistical Inferencer
        scanner = ScannerMotifs(self.genome, scan_width=self.scan_width)

        # Save the TF-only Metacells to disk for the Dashboard 
        self._save_metacells_to_parquet(scanner.all_tfs)

        inferencer = DynamicNetworkInference(self.metacell_dict, self.timepoints, self.time_var)
        
        # 3. Load Coordinates locally
        gene_coords_df = self._load_local_gene_annotations(bed_file_path, dropout_threshold)
        
        # Filter out target genes that happen to be TFs 
        gene_coords_df = gene_coords_df[~gene_coords_df['symbol'].isin(scanner.all_tfs)]

        # 4. Main Inference Loop
        logger.info("Starting TF-Gene Network Inference...")
        for _, row in gene_coords_df.iterrows():
            gene = row['symbol']
            chrom = row['chrom']
            if not chrom.startswith('chr'):
                chrom = f"chr{chrom}"
                
            start = int(row['start'])
            strand = row['strand_int']
            
            # Step 4a: Get Candidate TFs via Motif Scanning
            try:
                candidate_tfs = scanner.run_single_scan(chrom, start, strand)
            except Exception as e:
                logger.warning(f"Scanner error for {gene}: {e}")
                continue
                
            if not candidate_tfs:
                continue
                
            # Step 4b: Compute Time-Varying Edges
            inferencer.infer_tf_gene_edges(gene, candidate_tfs, dropout_threshold)

        # 5. Compute TF-TF Co-regulation
        logger.info("Starting TF-TF Co-regulation Inference...")
        inferencer.infer_tf_coregulation(scanner.all_tfs, self.save_path, dropout_threshold)

        # 6. Apply Consensus Layout & Save
        self._save_with_consensus_layout(inferencer.networks)
        logger.info("Pipeline Execution Complete!")

    def _save_with_consensus_layout(self, networks):
        """
        Calculates a stable (x,y) layout across all timepoints so nodes don't jump 
        around in the Dash visualization, then saves the networks.
        """
        logger.info("Computing consensus graph layout for stable visualization...")
        
        # Build an aggregate graph containing all edges across time
        consensus_graph = nx.Graph()
        for time, net in networks.items():
            for u, v, data in net.edges(data=True):
                weight = abs(data.get('coefficient', 0)) + abs(data.get('correlation', 0))
                if consensus_graph.has_edge(u, v):
                    consensus_graph[u][v]['weight'] += weight
                else:
                    consensus_graph.add_edge(u, v, weight=weight)
                    
        # Compute layout once (using circular_layout, but could use graphviz later maybe)
        pos = nx.circular_layout(consensus_graph)
        
        # Inject coordinates into individual networks and save
        for time in self.timepoints:
            net = networks[time]
            for node in net.nodes():
                if node in pos:
                    net.nodes[node]['x'] = float(pos[node][0])
                    net.nodes[node]['y'] = float(pos[node][1])
            
            save_file = os.path.join(self.save_path, f'network_{time}.gml.gz')
            nx.write_gml(net, save_file)
            
        logger.info(f"Saved layout-stabilized result graphs to {self.save_path}")
