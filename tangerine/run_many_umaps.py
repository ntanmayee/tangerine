# python
# qsub -cwd -V -b y -e qsub_outs/umap.txt -o qsub_outs/umap.txt python run_many_umaps.py

from find_motifs import Network
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm

def embed_pca(adata_full, time, selected_genes, pca_components=20):
    # filter by time
    obs_index = network.adata.obs.index.map(lambda x: time in x)
    # filter by highly expressed genes
    adata_full.var['in_list'] = adata_full.var.index.map(
        lambda value: value in selected_genes)
    adata = adata_full[obs_index, adata_full.var.in_list]
    
    pca = PCA(n_components=pca_components, svd_solver='arpack')
    
    # unit mean
    a = adata.X.toarray()
#     pca.fit(a - a.mean(axis=0))
    pca.fit(a)

    return pca.components_.T[:, 1:]

def make_relation(from_df, to_df):
    left = pd.DataFrame(data=np.arange(len(from_df)), index=from_df.index)
    right = pd.DataFrame(data=np.arange(len(to_df)), index=to_df.index)
    merge = pd.merge(left, right, left_index=True, right_index=True)
    return dict(merge.values)

def filter_adata(adata_full, time, selected_genes):
    # filter by time
    obs_index = network.adata.obs.index.map(lambda x: time in x)
    # filter by highly expressed genes
    adata_full.var['in_list'] = adata_full.var.index.map(
        lambda value: value in selected_genes)
    adata = adata_full[obs_index, adata_full.var.in_list]
    return adata

def axis_bounds(embedding):
    left, right = embedding.T[0].min(), embedding.T[0].max()
    bottom, top = embedding.T[1].min(), embedding.T[1].max()
    adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
    return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]

def run_single(random_state, iteration):
    aligned_mapper = umap.aligned_umap.AlignedUMAP(
        metric="euclidean",
        n_neighbors=50,
        alignment_regularisation=0.1,
        alignment_window_size=1,
        n_epochs=200,
        random_state=random_state,
    ).fit(data, relations=relations)
    embeddings = aligned_mapper.embeddings_

    pos_df = pd.DataFrame(index=selected_genes)

    for i, time in enumerate(timepoints):
        pos_df[f'{time}_x'] = aligned_mapper.embeddings_[i][:,0]
        pos_df[f'{time}_y'] = aligned_mapper.embeddings_[i][:,1]

    pos_df['is_tf'] = pos_df.index.map(lambda x: x in network.scanner.all_tfs)

    pos_df.to_csv(f'umap_results/aligned_umap_pos_skip_1_v{iteration}.csv')


if __name__ == '__main__':
    path_to_adata = '../../sc/ipynb/new_data_mar23/write/adata_network_inference.h5ad'
    timepoints = ['0h', '6h', '18h', '54h']
    dropout_threshold = 50
    num_iter = 100
    
    print('Making network object...', end='')
    network = Network(path_to_adata, timepoints)
    print('Done')

    selected_genes = list(network.adata.var[network.adata.var.pct_dropout_by_counts < dropout_threshold].index)
    print(f'Number of selected genes = {len(selected_genes)}')

    relations = [
        make_relation(filter_adata(network.adata, '0h', selected_genes).var, 
                      filter_adata(network.adata, '6h', selected_genes).var),
        make_relation(filter_adata(network.adata, '6h', selected_genes).var, 
                      filter_adata(network.adata, '18h', selected_genes).var), 
        make_relation(filter_adata(network.adata, '18h', selected_genes).var, 
                      filter_adata(network.adata, '54h', selected_genes).var)
    ]
    
    print('Embedding PCA...', end='')
    data = [embed_pca(network.adata, t, selected_genes) for t in timepoints]
    print('Done')
    
    generator = np.random.default_rng(seed=None)
    for i in tqdm(range(num_iter)):
        random_state = generator.integers(low=1, high=123899)
        run_single(random_state, i)
