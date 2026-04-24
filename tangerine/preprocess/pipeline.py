import os
import pandas as pd
import scanpy as sc
import networkx as nx
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sps
import numpy as np
import json

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

    def _save_config(self):
        config = {
            'adata_path': self.adata_path,
            'timepoints': self.timepoints,
            'genome': self.genome,
            'time_var': self.time_var,
            'save_path': self.save_path,
            'scan_width': self.scan_width,
            'metacell_method': self.metacell_method,
            'n_pcs': self.n_pcs,
            'n_neighbors': self.n_neighbors
        }
        save_file = os.path.join(self.save_path, 'config.json')
        with open(save_file, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info('Saved run parameters to config.json')

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


    def _save_metacells_to_parquet(self, genes_to_save, chunk_size=2000):
        """
        Filters the generated metacells to include TFs and inference genes.
        Safely handles both sparse and dense matrices and removes duplicate genes.
        """
        logger.info("Saving metacell expression to Parquet via chunked processing...")
        
        # FIX 1: Strip duplicates from the incoming list (BED files cause duplicates)
        # Using dict.fromkeys() removes duplicates while preserving the original list order
        unique_genes_to_save = list(dict.fromkeys(genes_to_save))
        
        for time, m_data in self.metacell_dict.items():
            file_name = f"metacells_{time}.parquet"
            save_file = os.path.join(self.save_path, file_name)

            if isinstance(m_data, sc.AnnData):
                # Pre-slice the matrix 
                valid_genes = [g for g in unique_genes_to_save if g in m_data.var_names]
                sliced_adata = m_data[:, valid_genes]

                X = sliced_adata.X
                
                # FIX 2: Safely check if the matrix is sparse before calling sparse methods
                is_sparse = sps.issparse(X)
                if is_sparse and not sps.isspmatrix_csr(X):
                    X = X.tocsr()

                n_cells = X.shape[0]
                writer = None

                # Iterate through the matrix in chunks of rows (cells)
                for start_idx in range(0, n_cells, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_cells)

                    # FIX 3: Extract chunk based on whether it is sparse or dense
                    if is_sparse:
                        chunk_dense = X[start_idx:end_idx, :].toarray()
                    else:
                        chunk_dense = X[start_idx:end_idx, :]

                    # Convert to a temporary DataFrame, then to a PyArrow Table
                    chunk_df = pd.DataFrame(chunk_dense, columns=valid_genes)
                    table = pa.Table.from_pandas(chunk_df)

                    # Initialize the Parquet writer on the first chunk
                    if writer is None:
                        writer = pq.ParquetWriter(save_file, table.schema)
                    
                    # Append the chunk to the file
                    writer.write_table(table)

                # Close the file connection
                if writer:
                    writer.close()

                logger.info(f"Saved {file_name} with {len(valid_genes)} genes across {n_cells} cells.")

            else:
                # Fallback: If it's already a dense Pandas DataFrame, just filter and save
                df = pd.DataFrame(m_data)
                valid_genes = [g for g in unique_genes_to_save if g in df.columns]
                df[valid_genes].to_parquet(save_file)
                
                logger.info(f"Saved {file_name} with {len(valid_genes)} genes (Dense fallback).")

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
        
        # Save parameters to json file before starting pipeline
        self._save_config()
        
        # 1. Prepare Data
        self.load_and_preprocess_data()
        self.generate_metacells()
        
        # 2. Initialize Prior Scanner & Statistical Inferencer
        scanner = ScannerMotifs(self.genome, scan_width=self.scan_width)

        inferencer = DynamicNetworkInference(self.metacell_dict, self.timepoints, self.time_var)
        
        # 3. Load Coordinates locally
        gene_coords_df = self._load_local_gene_annotations(bed_file_path, dropout_threshold)
        
        # Filter out target genes that happen to be TFs 
        gene_coords_df = gene_coords_df[~gene_coords_df['symbol'].isin(scanner.all_tfs)]

        # make list of genes to save metacell data later
        good_genes = []

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

            # Append to list
            good_genes.append(gene)

        # 5. Compute TF-TF Co-regulation
        logger.info("Starting TF-TF Co-regulation Inference...")
        inferencer.infer_tf_coregulation(scanner.all_tfs, self.save_path, dropout_threshold)

        # Save the TF-only Metacells to disk for the Dashboard 
        self._save_metacells_to_parquet(scanner.all_tfs + good_genes)

        # 6. Apply Consensus Layout & Save
        self._save_networks(inferencer.networks)
        logger.info("Pipeline Execution Complete!")

    def _save_networks(self, networks):
        """
        Saves the inferred TF-Gene bipartite networks to disk.
        Spatial (x,y) layouts are handled dynamically by the frontend Cytoscape engine.
        """
        logger.info("Saving topological networks to GML...")
        
        for time in self.timepoints:
            net = networks[time]
            save_file = os.path.join(self.save_path, f'network_{time}.gml.gz')
            nx.write_gml(net, save_file)
            
        logger.info(f"Saved result graphs to {self.save_path}")
