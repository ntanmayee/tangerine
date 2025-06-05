from find_motifs import Network
from args import parser

if __name__ == "__main__":
    # path_to_adata = '../../sc/ipynb/new_data_mar23/write/adata_network_inference.h5ad'
    # timepoints = ['0h', '6h', '18h', '54h']

    # network = Network(path_to_adata, timepoints)
    # # network.run_all_genes('arid1a_networks')
    # network.run_tf_correlation('arid1a_networks')

    args = parser.parse_args()

    # path_to_adata = '/cluster/gs_lab/tnarendra001/sc/ipynb/smart_seq/smart_seq_norm.h5ad'
    # timepoints = ['0h', '2h', '6h', '54h']

    network = Network(args.adata_path, args.timepoints, args.genome, time_var=args.time_var, scan_width=args.scan_width)
    network.run_tf_correlation(args.save_name)
    network.run_all_genes(args.save_name)
