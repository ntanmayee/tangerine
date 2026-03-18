import pandas as pd
import numpy as np
import scanpy as sc
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from tangerine.preprocess.logger import logger

class MetacellGenerator(ABC):
    """
    Abstract base class for metacell generation strategies.
    """
    def __init__(self, adata, time_var, timepoints, min_cells_for_metacell=200, cells_per_metacell=20):
        self.adata = adata
        self.time_var = time_var
        self.timepoints = timepoints
        self.min_cells_for_metacell = min_cells_for_metacell
        self.cells_per_metacell = cells_per_metacell

    @abstractmethod
    def generate(self) -> dict:
        """
        Executes the metacell aggregation per timepoint.
        
        Returns:
            dict: A dictionary mapping each timepoint (key) to an 
                  aggregated AnnData object (value).
        """
        pass


class KMeansMetacellGenerator(MetacellGenerator):
    """
    Native implementation of Metacells using K-Means clustering.
    """
    def generate(self) -> dict:
        logger.info("Starting Native Metacell Aggregation (K-Means)...")
        metacell_cache = {}
        
        for time in self.timepoints:
            subset = self.adata[self.adata.obs[self.time_var] == time].copy()
            n_cells = subset.n_obs
            
            # Fallback to raw cells if there isn't enough data
            if n_cells < self.min_cells_for_metacell:
                logger.warning(f"Timepoint {time}: {n_cells} cells < threshold ({self.min_cells_for_metacell}). Using RAW cells.")
                metacell_cache[time] = subset
                continue
                
            n_metacells = max(2, int(n_cells / self.cells_per_metacell))
            logger.info(f"Timepoint {time}: Aggregating {n_cells} cells -> {n_metacells} Metacells.")
            
            try:
                # Ensure PCA exists for clustering
                if 'X_pca' not in subset.obsm.keys():
                    sc.tl.pca(subset)
                
                # Perform Clustering
                kmeans = KMeans(n_clusters=n_metacells, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(subset.obsm['X_pca'])
                subset.obs['metacell_id'] = clusters.astype(str)
                
                # Aggregate Counts (Sum)
                if subset.raw is not None:
                    raw_df = pd.DataFrame(
                        subset.raw.X.toarray() if hasattr(subset.raw.X, 'toarray') else subset.raw.X, 
                        index=subset.obs_names, 
                        columns=subset.raw.var_names
                    )
                else:
                    data_mat = subset.X.toarray() if hasattr(subset.X, 'toarray') else subset.X
                    if data_mat.max() < 20: 
                        data_mat = np.expm1(data_mat) # Crude un-log
                    raw_df = pd.DataFrame(data_mat, index=subset.obs_names, columns=subset.var_names)
                
                # Group by Metacell ID and Sum
                aggregated_df = raw_df.groupby(subset.obs['metacell_id']).sum()
                
                # Create New AnnData for this timepoint
                meta_adata = sc.AnnData(aggregated_df)
                meta_adata.obs[self.time_var] = time 
                
                # Normalize the new Metacells
                sc.pp.normalize_total(meta_adata, target_sum=1e4)
                sc.pp.log1p(meta_adata)
                
                metacell_cache[time] = meta_adata
                
            except Exception as e:
                logger.error(f"Metacell aggregation failed for {time}: {e}. Fallback to Raw.")
                metacell_cache[time] = subset
                
        return metacell_cache


class SEACellsMetacellGenerator(MetacellGenerator):
    """
    Implementation of Metacells using the SEACells algorithm.
    Requires 'SEACells' to be installed via pip/conda.
    """
    def generate(self) -> dict:
        logger.info("Starting SEACells Metacell Aggregation...")
        
        try:
            import SEACells
        except ImportError:
            logger.error("SEACells is not installed. Please run `pip install SEACells` or use KMeans.")
            raise

        metacell_cache = {}
        
        for time in self.timepoints:
            subset = self.adata[self.adata.obs[self.time_var] == time].copy()
            n_cells = subset.n_obs
            
            if n_cells < self.min_cells_for_metacell:
                logger.warning(f"Timepoint {time}: Not enough cells for SEACells. Using RAW cells.")
                metacell_cache[time] = subset
                continue
            
            n_metacells = max(2, int(n_cells / self.cells_per_metacell))
            logger.info(f"Timepoint {time}: Computing {n_metacells} SEACells.")
            
            try:
                build_kernel_on = 'X_pca' 
                n_waypoint_eigs = 10
                model = SEACells.core.SEACells(subset, build_kernel_on=build_kernel_on, 
                                               n_SEACells=n_metacells, n_waypoint_eigs=n_waypoint_eigs, 
                                               convergence_epsilon = 1e-5)
                model.construct_kernel_matrix()
                model.initialize_archetypes()
                model.fit(min_iter=10, max_iter=50)

                # Extract Aggregated Counts ---
                # SEACells returns a DataFrame mapping cell index to a 'SEACell' ID
                assignments = model.get_hard_assignments()
                subset.obs['metacell_id'] = assignments['SEACell']

                # Extract raw data safely 
                if subset.raw is not None:
                    data_mat = subset.raw.X.toarray() if hasattr(subset.raw.X, 'toarray') else subset.raw.X
                    raw_df = pd.DataFrame(data_mat, index=subset.obs_names, columns=subset.raw.var_names)
                else:
                    data_mat = subset.X.toarray() if hasattr(subset.X, 'toarray') else subset.X
                    if data_mat.max() < 20: 
                        data_mat = np.expm1(data_mat) # Crude un-log
                    raw_df = pd.DataFrame(data_mat, index=subset.obs_names, columns=subset.var_names)
                
                # Group by the SEACell ID and sum the counts
                aggregated_df = raw_df.groupby(subset.obs['metacell_id']).sum()

                # Create meta_adata, normalize, log1p 
                meta_adata = sc.AnnData(aggregated_df)
                
                # Restore the time label
                meta_adata.obs[self.time_var] = time 
                
                # Record how many raw cells make up each SEACell
                cell_counts = subset.obs['metacell_id'].value_counts()
                meta_adata.obs['n_cells'] = cell_counts[meta_adata.obs_names].values

                # Standardize the new metacells
                sc.pp.normalize_total(meta_adata, target_sum=1e4)
                sc.pp.log1p(meta_adata)
                
                # Save to cache
                metacell_cache[time] = meta_adata
                
            except Exception as e:
                logger.error(f"SEACells aggregation failed for {time}: {e}. Fallback to Raw.")
                metacell_cache[time] = subset

        return metacell_cache
