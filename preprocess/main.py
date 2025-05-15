from find_motifs import Network

if __name__ == "__main__":
    # path_to_adata = '../../sc/ipynb/new_data_mar23/write/adata_network_inference.h5ad'
    # timepoints = ['0h', '6h', '18h', '54h']

    # network = Network(path_to_adata, timepoints)
    # # network.run_all_genes('arid1a_networks')
    # network.run_tf_correlation('arid1a_networks')

    path_to_adata = '/cluster/gs_lab/tnarendra001/sc/ipynb/smart_seq/smart_seq_norm.h5ad'
    timepoints = ['0h', '2h', '6h', '54h']

    network = Network(path_to_adata, timepoints)
    network.run_tf_correlation('smartseq_networks')
