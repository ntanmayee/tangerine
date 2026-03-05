from gimmemotifs.scanner import Scanner
from gimmemotifs.motif import read_motifs
from gimmemotifs.config import MotifConfig
import scanpy as sc
from tangerine.preprocess.utils import peak2fasta, scan_dna_for_motifs
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
from tangerine.preprocess.logger import logger
from sklearn.cluster import KMeans


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
    """Class to scan motifs in specified region of genome
    """
    def __init__(self, genome, n_cpus=1, fpr=0.02, scan_width=5000) -> None:
        """Constructor for ScannerMotifs class

        Args:
            genome (str): Genome which will be used for scanning (currently supports mm10 and hg19)
            n_cpus (int, optional): Number of CPU cores to use. Defaults to 1.
            fpr (float, optional): False positive rate. Defaults to 0.02.
            scan_width (int, optional): Width of genomic region upstream of promoter to scan (in bp). Defaults to 5000.
        """
        self.genome = genome

        self.scanner_object = Scanner(ncpus=n_cpus)
        self.motifs = default_motifs()

        self.scanner_object.set_background(genome=genome)
        self.scanner_object.set_motifs(self.motifs)
        self.scanner_object.set_threshold(fpr=fpr)

        self.scan_width = scan_width
        self.all_tfs = self.get_all_motifs()

    def get_all_motifs(self):
        """Makes a consolidated set of TFs in the database

        Returns:
            list: List of consolidated TFs in the database
        """
        all_tfs = []

        for motif in self.motifs:
            all_tfs.extend(motif.factors['direct'])
            all_tfs.extend(motif.factors['indirect\nor predicted'])

        all_tfs = [a.capitalize() for a in all_tfs]
        all_tfs = list(set(all_tfs))
        return all_tfs

    def scan_motifs_single(self, chr_name, start, strand):
        """Scan for TF motifs upstream of a single gene

        Args:
            chr_name (str): Chromosome name
            start (int): start position in bp
            strand (int): strandedness of the gene  

        Returns:
            pandas DataFrame: Results of scan in dataframe
        """
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
        """Run single scan and return list of TFs found

        Args:
            chr_name (str): Chromosome name
            start (int): start position
            strand (int): strandedness (-1 or 1)
            score (float, optional): Threshold score to filter. Defaults to None.
            include_indirect (bool, optional): Whether or not to include indirect TF binding sites. Defaults to False.

        Returns:
            list: List of TFs whose motif was found in the region
        """
        scanned_df = self.scan_motifs_single(chr_name, start, strand)
        tf_list = self.get_filtered_tfs(scanned_df, score, include_indirect)
        return tf_list


class Network(object):
    """Class to estimate TF-TF and TF-gene networks
    """
    def __init__(self, path_to_adata, timepoints, genome, time_var, n_pcs=35, n_neighbors=100, scan_width=10000, use_metacells=True, min_cells_for_metacell=200, cells_per_metacell=50) -> None:
        """Constructor for Network class

        Args:
            path_to_adata (str): Path to adata object
            timepoints (list): List of timepoints to analyse
            genome (str): Genome of organism
            time_var (str): Column name in adata.obs denoting time
            n_pcs (int, optional): Number of principal components to use. Defaults to 35.
            n_neighbors (int, optional): Number of neighbours for KNN graph. Defaults to 100.
            scan_width (int, optional): Width of scanning window for motifs. Defaults to 10000.
            use_metacells (bool): If True, tries to aggregate cells to fix dropout.
            min_cells_for_metacell (int): Threshold. If n_cells < this, use RAW data.
            cells_per_metacell (int): Target grain size (e.g., 50 cells -> 1 metacell).

        Raises:
            NotImplementedError: If unknown genome is given, raises this error
        """
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

        self.use_metacells = use_metacells
        self.min_cells_for_metacell = min_cells_for_metacell
        self.cells_per_metacell = cells_per_metacell
        self.metacell_cache = {}
        
        self.preprocess_adata()

        if self.use_metacells:
            self.generate_native_metacells()

        self.scanner = ScannerMotifs(self.genome, scan_width=scan_width)

    def __str__(self) -> str:
        print_str = f'Tangerine Network object with timepoints {str(self.timepoints)}\n\n{str(self.adata)}'
        return print_str
    
    def __repr__(self) -> str:
        return self.__str__()

    def preprocess_adata(self):
        """Read and preprocess AnnData object

        Raises:
            NotImplementedError: Raised if unable to read single cell data
        """
        if self.path_to_adata.endswith('.h5ad'):
            self.adata = sc.read_h5ad(self.path_to_adata)
            logger.info('Succesfully read in adata')
        else:
            raise NotImplementedError('Cannot read single cell data file.')
        
        # check if pca and umap already exists, otherwise preprocess
        sc.pp.calculate_qc_metrics(self.adata, inplace=True)

        try:
            if 'pca' not in self.adata.uns.keys() or 'umap' not in self.adata.uns.keys():
                # count normalization to 10,000 reads per cell 
                sc.pp.normalize_total(self.adata, target_sum=1e4)
                self.adata.raw = self.adata.copy()

                sc.pp.log1p(self.adata)
                sc.tl.pca(self.adata, svd_solver='arpack')
                sc.pp.neighbors(self.adata, n_pcs=self.n_pcs, metric='euclidean', n_neighbors=self.n_neighbors)
                sc.tl.umap(self.adata)
        except:
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            self.adata.raw = self.adata.copy()

            sc.pp.log1p(self.adata)
            sc.tl.pca(self.adata, svd_solver='arpack')
            sc.pp.neighbors(self.adata, n_pcs=self.n_pcs, metric='euclidean', n_neighbors=self.n_neighbors)
            sc.tl.umap(self.adata)
    
    def generate_native_metacells(self):
        """Implementation of Metacells using K-Means"""
        logger.info("Starting Native Metacell Aggregation...")
        
        for time in self.timepoints:
            subset = self.adata[self.adata.obs[self.time_var] == time].copy()
            n_cells = subset.n_obs
            
            if n_cells < self.min_cells_for_metacell:
                logger.warning(f"Timepoint {time}: {n_cells} cells < threshold ({self.min_cells_for_metacell}). Using RAW cells.")
                self.metacell_cache[time] = subset
                continue
                
            n_metacells = max(2, int(n_cells / self.cells_per_metacell))
            logger.info(f"Timepoint {time}: Aggregating {n_cells} cells -> {n_metacells} Metacells (K-Means).")
            
            try:
                if 'X_pca' not in subset.obsm.keys():
                    sc.tl.pca(subset)
                
                kmeans = KMeans(n_clusters=n_metacells, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(subset.obsm['X_pca'])
                subset.obs['metacell_id'] = clusters.astype(str)
                
                # Aggregate Counts (Sum)
                # We group by the new cluster ID and sum the raw counts
                # Note: Assuming subset.X contains log-normalized data, we should ideally sum RAW counts.
                # If .raw exists, use it. Else, un-log if possible or sum normalized (less ideal but workable).
                
                if subset.raw is not None:
                    raw_df = pd.DataFrame(subset.raw.X.toarray() if hasattr(subset.raw.X, 'toarray') else subset.raw.X, 
                                          index=subset.obs_names, columns=subset.raw.var_names)
                else:
                    # Fallback: Use .X (warn user if it looks logged)
                    data_mat = subset.X.toarray() if hasattr(subset.X, 'toarray') else subset.X
                    if data_mat.max() < 20: 
                        # Crude check for log data. Exponentiate to get approx counts back.
                        data_mat = np.expm1(data_mat)
                    raw_df = pd.DataFrame(data_mat, index=subset.obs_names, columns=subset.var_names)
                
                # Group by Metacell ID and Sum
                aggregated_df = raw_df.groupby(subset.obs['metacell_id']).sum()
                
                # Create New AnnData
                meta_adata = sc.AnnData(aggregated_df)
                meta_adata.obs[self.time_var] = time # Restore time label
                
                # Normalize the new Metacells
                sc.pp.normalize_total(meta_adata, target_sum=1e4)
                sc.pp.log1p(meta_adata)
                
                self.metacell_cache[time] = meta_adata
                
            except Exception as e:
                logger.error(f"Metacell aggregation failed for {time}: {e}. Fallback to Raw.")
                self.metacell_cache[time] = subset

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
        """
        Run network inference for a single gene.
        Updates:
        - Uses 'Union' filtering: Candidate TFs are kept if they pass dropout in ANY time point.
        - Fills NaN correlations with 0.
        - Skips Ridge regression for TFs that are completely silenced in a specific time point.
        """
        assert gene_name in self.adata.var.index, f'Gene {gene_name} not found in self.adata'
        assert chr_name not in self.scanner.all_tfs, 'Target gene cannot be a transcription factor'

        # 1. Motif Scan to get initial candidates
        try:
            raw_tf_list = self.scanner.run_single_scan(chr_name, start, strand, score, include_indirect)
        except Exception as e:
            logger.warning(f'Scanner error for gene {gene_name}, {chr_name}: {e}')
            return pd.DataFrame(), pd.DataFrame()

        if len(raw_tf_list) < 1:
            return None, None

        # 2. Relaxed Filtering (Union of Time-Specific Leaders)
        # We check if a TF passes the dropout threshold in AT LEAST ONE time point.
        valid_tfs_set = set()
        
        # Pre-check which TFs are in the adata object at all
        possible_tfs = [tf for tf in raw_tf_list if tf in self.adata.var_names]
        
        if not possible_tfs:
             return None, None

        for time in self.timepoints:
            subset = self.adata[self.adata.obs[self.time_var] == time]
            
            # Get subset of data for these TFs
            # Using sparse check for speed
            X_subset = subset[:, possible_tfs].X
            n_cells = X_subset.shape[0]
            
            if hasattr(X_subset, "getnnz"):
                n_expressed = X_subset.getnnz(axis=0)
            else:
                n_expressed = (X_subset != 0).sum(axis=0)
                
            pct_dropout = 100 * (1 - (n_expressed / n_cells))
            
            # Keep TFs that pass threshold in this specific time point
            passing_indices = np.where(pct_dropout <= dropout_threshold)[0]
            passed_tfs = [possible_tfs[i] for i in passing_indices]
            valid_tfs_set.update(passed_tfs)
            
        final_tf_list = list(valid_tfs_set)
        
        if len(final_tf_list) < 1:
            return None, None

        # 3. Initialize Results with 0.0 (Handles silenced factors automatically)
        correlation_df = pd.DataFrame(0.0, index=final_tf_list, columns=self.timepoints)    
        coefficients_df = pd.DataFrame(0.0, index=final_tf_list, columns=self.timepoints)

        # 4. Compute Stats Per Timepoint
        for time in self.timepoints:
            # Handle Metacells if enabled
            if self.use_metacells and time in self.metacell_cache:
                current_adata = self.metacell_cache[time]
            else:
                current_adata = self.adata[self.adata.obs[self.time_var] == time]

            # Columns to fetch: Gene + Valid TFs that exist in this time point
            valid_cols = [g for g in ([gene_name] + final_tf_list) if g in current_adata.var_names]
            
            if gene_name not in valid_cols:
                continue

            data_df = sc.get.obs_df(current_adata, valid_cols, use_raw=False)

            if len(data_df) < 5: 
                continue
            
            # --- A. Correlation ---
            if gene_name in data_df.columns:
                corr_series = data_df.corr(method='spearman')[gene_name]
                # FIX: Fill NaNs (from 0-variance TFs) with 0.0
                corr_series = corr_series.fillna(0.0)
                
                common_tfs = corr_series.index.intersection(final_tf_list)
                correlation_df.loc[common_tfs, time] = corr_series[common_tfs]
            
            # --- B. Ridge Regression ---
            # Filter predictors: Only use TFs available in this batch
            available_predictors = [tf for tf in final_tf_list if tf in valid_cols]
            
            if len(available_predictors) > 0:
                X = data_df[available_predictors]
                y = data_df[gene_name]
                
                # CRITICAL: Identify "Active" TFs for this specific time point
                # If a TF is all 0s here, we must NOT regress on it (it's noise/singular).
                # We leave its coefficient as 0.0 (from initialization).
                active_tfs = X.columns[(X != 0).any(axis=0)].tolist()
                
                if len(active_tfs) > 0:
                    X_subset = X[active_tfs]
                    
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.1, random_state=29)

                        model = Ridge()
                        model.fit(X_train, y_train)
                        
                        # Store coefficients ONLY for the active TFs
                        for i, tf in enumerate(active_tfs):
                            coefficients_df.loc[tf, time] = float(model.coef_[i])
                            
                        # Store model score on the gene node
                        network = self.networks[time]
                        nx.set_node_attributes(network, {gene_name: {'score': float(model.score(X_test, y_test))}})
                        
                        # Add edges to graph
                        for tf in active_tfs:
                            corr = float(correlation_df.loc[tf, time])
                            coef = float(coefficients_df.loc[tf, time])
                            
                            # Optional: Filter tiny edges to save space? 
                            # Currently keeping all to match original logic, but casting to float.
                            network.add_edge(tf, gene_name, correlation=corr, coefficient=coef)
                            
                    except Exception as e:
                        logger.warning(f"Regression failed for {gene_name} at {time}: {e}")
                        continue

        return correlation_df, coefficients_df
    
    def check_path_exists(self, path_name):
        """Check if given path exists, if not create it

        Args:
            path_name (str): path
        """
        path = pathlib.Path(path_name)
        path.mkdir(parents=True, exist_ok=True)

    def run_all_genes(self, save_path, dropout_threshold=50):
        """Run motif search for all genes

        Args:
            save_path (str): Path to save results
            dropout_threshold (int, optional): Maximum dropout value for genes. Defaults to 50.
        """
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
            """Mini function to use pandas apply function

            Args:
                row: Row from pandas DataFrame

            Returns:
                list: list of TF whose motif was found
            """
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
        """_summary_

        Args:
            save_path (_type_): _description_
            dropout_threshold (int, optional): _description_. Defaults to 50.

        Returns:
            _type_: _description_
        """
        self.check_path_exists(save_path)

        logger.info('Filtering TFs based on time-specific dropout...')
        
        # 1. Start with all possible TFs from the scanner
        all_candidates = self.scanner.all_tfs
        valid_tfs = set()

        for time in self.timepoints:
            # 2. Get the subset of raw cells for this time point
            subset = self.adata[self.adata.obs[self.time_var] == time]
            
            # 3. Find which candidate TFs are actually in the dataset
            present_tfs = [tf for tf in all_candidates if tf in subset.var_names]
            
            if not present_tfs:
                continue

            # 4. Calculate dropout specifically for this time point
            X_subset = subset[:, present_tfs].X
            n_cells = X_subset.shape[0]

            if hasattr(X_subset, "getnnz"):
                # Sparse matrix: getnnz(0) counts non-zero elements per column
                n_expressed = X_subset.getnnz(axis=0)
                pct_dropout = 100 * (1 - (n_expressed / n_cells))
            else:
                # Dense matrix (fallback)
                n_zeros = (X_subset == 0).sum(axis=0)
                pct_dropout = 100 * (n_zeros / n_cells)

            # 5. Identify TFs that are good quality (>50% expression) in THIS time point
            # np.where returns indices, so we map them back to TF names
            passing_indices = np.where(pct_dropout <= dropout_threshold)[0]
            passed_tfs = [present_tfs[i] for i in passing_indices]
            
            # 6. Add to the "Union" set
            valid_tfs.update(passed_tfs)

        filt_tfs = list(valid_tfs)
        logger.info(f'Number of filtered TFs (Union across time): {len(filt_tfs)}')

        # def filter_selected_tfs(tf_list, dropout_threshold=50):
        #     filtered_tfs = []

        #     for tf in tf_list:
        #         try:
        #             dropout = self.adata.var.loc[tf]['pct_dropout_by_counts'] 
        #             if dropout <= dropout_threshold:
        #                 filtered_tfs.append(tf)
        #         except KeyError:
        #             pass

        #         try:
        #             dropout = self.adata.var.loc[tf.upper()]['pct_dropout_by_counts'] 
        #             if dropout <= dropout_threshold:
        #                 filtered_tfs.append(tf)
        #         except KeyError:
        #             pass
            
        #     return filtered_tfs

        # filt_tfs = filter_selected_tfs(self.scanner.all_tfs, dropout_threshold)
        # logger.info(f'Number of filtered TFs: {len(filt_tfs)}')

        # compute correlation
        logger.info('Computing correlation among TFs...')
        corr_dfs = {}
        for time in self.timepoints:
            data_df = sc.get.obs_df(self.adata, filt_tfs + [self.time_var], use_raw=False)
            data_df = data_df[data_df[self.time_var] == time].drop(self.time_var, axis=1)
            corr_dfs[time] = data_df.corr('spearman').fillna(0)

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
