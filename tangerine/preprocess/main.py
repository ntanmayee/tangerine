from find_motifs import Network
from args import parser

if __name__ == "__main__":
    args = parser.parse_args()

    network = Network(args.adata_path, args.timepoints, args.genome, time_var=args.time_var, scan_width=args.scan_width)
    network.run_tf_correlation(args.save_name)
    network.run_all_genes(args.save_name)
