import pandas as pd
import numpy as np
import scanpy as sc
import networkx as nx
from os.path import join
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from tangerine.preprocess.logger import logger


class DynamicNetworkInference:
    """
    Handles the statistical inference of the time-varying Gene Regulatory Network.
    Calculates TF-Gene (Ridge Regression) and TF-TF (Spearman) dynamics.
    """
    def __init__(self, metacell_dict, timepoints, time_var):
        """
        Args:
            metacell_dict (dict): Dictionary mapping timepoints to AnnData objects.
            timepoints (list): List of timepoints to analyze.
            time_var (str): Column name in adata.obs denoting time.
        """
        self.metacell_dict = metacell_dict
        self.timepoints = timepoints
        self.time_var = time_var
        
        # Initialize an empty directed graph for each timepoint
        self.networks = {t: nx.DiGraph() for t in self.timepoints}

    def infer_tf_gene_edges(self, gene_name, candidate_tfs, dropout_threshold=50):
        """
        Runs network inference for a single gene against a prior list of candidate TFs.
        
        Args:
            gene_name (str): Target gene.
            candidate_tfs (list): List of TFs whose motifs were found upstream.
            dropout_threshold (int): Max dropout percentage to keep a TF.
            
        Returns:
            tuple: (correlation_df, coefficients_df)
        """
        # 1. Relaxed Filtering (Union of Time-Specific Leaders)
        valid_tfs_set = set()
        
        for time in self.timepoints:
            current_adata = self.metacell_dict[time]
            possible_tfs = [tf for tf in candidate_tfs if tf in current_adata.var_names]
            
            if not possible_tfs:
                 continue

            X_subset = current_adata[:, possible_tfs].X
            n_cells = X_subset.shape[0]
            
            if hasattr(X_subset, "getnnz"):
                n_expressed = X_subset.getnnz(axis=0)
            else:
                n_expressed = (X_subset != 0).sum(axis=0)
                
            pct_dropout = 100 * (1 - (n_expressed / n_cells))
            passing_indices = np.where(pct_dropout <= dropout_threshold)[0]
            passed_tfs = [possible_tfs[i] for i in passing_indices]
            valid_tfs_set.update(passed_tfs)
            
        final_tf_list = list(valid_tfs_set)
        
        if len(final_tf_list) < 1:
            return None, None

        # 2. Initialize Results with 0.0
        correlation_df = pd.DataFrame(0.0, index=final_tf_list, columns=self.timepoints)    
        coefficients_df = pd.DataFrame(0.0, index=final_tf_list, columns=self.timepoints)

        # 3. Compute Stats Per Timepoint
        for time in self.timepoints:
            current_adata = self.metacell_dict[time]
            valid_cols = [g for g in ([gene_name] + final_tf_list) if g in current_adata.var_names]
            
            if gene_name not in valid_cols:
                continue

            data_df = sc.get.obs_df(current_adata, valid_cols, use_raw=False)

            if len(data_df) < 5: 
                continue
            
            # --- A. Correlation ---
            if gene_name in data_df.columns:
                corr_series = data_df.corr(method='spearman')[gene_name].fillna(0.0)
                common_tfs = corr_series.index.intersection(final_tf_list)
                correlation_df.loc[common_tfs, time] = corr_series[common_tfs]
            
            # --- B. Ridge Regression ---
            available_predictors = [tf for tf in final_tf_list if tf in valid_cols]
            
            if len(available_predictors) > 0:
                X = data_df[available_predictors]
                y = data_df[gene_name]
                
                # Identify "Active" TFs for this specific time point
                active_tfs = X.columns[(X != 0).any(axis=0)].tolist()
                
                if len(active_tfs) > 0:
                    X_subset = X[active_tfs]
                    
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.1, random_state=29)

                        model = Ridge()
                        model.fit(X_train, y_train)
                        
                        # Store coefficients
                        for i, tf in enumerate(active_tfs):
                            coefficients_df.loc[tf, time] = float(model.coef_[i])
                            
                        # Store model score on the gene node
                        network = self.networks[time]
                        nx.set_node_attributes(network, {gene_name: {'score': float(model.score(X_test, y_test))}})
                        
                        # Add edges to graph
                        for tf in active_tfs:
                            corr = float(correlation_df.loc[tf, time])
                            coef = float(coefficients_df.loc[tf, time])
                            network.add_edge(tf, gene_name, correlation=corr, coefficient=coef)
                            
                    except Exception as e:
                        logger.warning(f"Regression failed for {gene_name} at {time}: {e}")
                        continue

        return correlation_df, coefficients_df

    def infer_tf_coregulation(self, all_candidate_tfs, save_path, dropout_threshold=50):
        """
        Computes TF-TF correlation matrices, orders them independently per timepoint, 
        and clusters them dynamically to track module evolution.
        """
        import os
        os.makedirs(save_path, exist_ok=True)

        logger.info('Filtering TFs based on time-specific dropout (Union strategy)...')
        valid_tfs = set()

        for time in self.timepoints:
            current_adata = self.metacell_dict[time]
            present_tfs = [tf for tf in all_candidate_tfs if tf in current_adata.var_names]
            
            if not present_tfs:
                continue

            X_subset = current_adata[:, present_tfs].X
            n_cells = X_subset.shape[0]

            if hasattr(X_subset, "getnnz"):
                n_expressed = X_subset.getnnz(axis=0)
            else:
                n_expressed = (X_subset != 0).sum(axis=0)
                
            pct_dropout = 100 * (1 - (n_expressed / n_cells))
            passing_indices = np.where(pct_dropout <= dropout_threshold)[0]
            passed_tfs = [present_tfs[i] for i in passing_indices]
            valid_tfs.update(passed_tfs)

        filt_tfs = list(valid_tfs)
        logger.info(f'Number of filtered TFs (Union across time): {len(filt_tfs)}')

        if len(filt_tfs) < 2:
            logger.error("Not enough TFs passed the dropout filter to perform clustering.")
            return

        # 1. Compute Correlation
        logger.info('Computing correlation among TFs...')
        corr_dfs = {}
        for time in self.timepoints:
            current_adata = self.metacell_dict[time]
            valid_cols = [col for col in filt_tfs if col in current_adata.var_names]
            
            # Extract data and compute correlation
            data_df = sc.get.obs_df(current_adata, valid_cols, use_raw=False)
            corr = data_df.corr('spearman')
            
            # Ensure consistent index/columns across all timepoints before clustering
            # (Fills missing TFs that dropped out in this specific timepoint with 0.0)
            corr = corr.reindex(index=filt_tfs, columns=filt_tfs).fillna(0.0)
            corr_dfs[time] = corr

        # 2. Independently Order, Save, and Cluster ALL timepoints
        logger.info('Clustering and ordering correlation matrices independently per timepoint...')
        cluster_df = pd.DataFrame(index=filt_tfs, columns=self.timepoints)

        for time in self.timepoints:
            current_corr = corr_dfs[time]
            X = current_corr.values
            
            # Compute linkage for the CURRENT timepoint
            Z = linkage(X, method='ward')
            
            # Determine optimal visual ordering for the CURRENT timepoint
            optimal_order_indices = leaves_list(Z)
            ordered_tfs = current_corr.index[optimal_order_indices]
            
            # Reorder the correlation matrix and save
            ordered_corr = current_corr.loc[ordered_tfs, ordered_tfs]
            ordered_corr.to_csv(join(save_path, f'tf_tf_{time}.csv'))

            # Compute dynamic clusters for the CURRENT timepoint
            distances = Z[:, 2]
            diffs = np.diff(distances)

            if len(diffs) > 0:
                max_gap_idx = np.argmax(diffs)
                num_clusters = max(2, len(distances) - max_gap_idx)
            else:
                num_clusters = 2
            
            # cluster_labels maps 1-to-1 with the original unordered current_corr.index
            cluster_labels = fcluster(Z, t=num_clusters, criterion='maxclust')
            cluster_df.loc[current_corr.index, time] = cluster_labels

        # 3. Format and Save Cluster Assignments
        for time in self.timepoints:
            cluster_df[time] = cluster_df[time].apply(lambda x: f'{time}-{int(x) if pd.notnull(x) else 0}')

        cluster_df.to_csv(join(save_path, 'louvain_tf.csv'))
        logger.info('Saved independently ordered matrices and dynamic clustering results to file.')
