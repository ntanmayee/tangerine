import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ap', '--adata_path', help='path to processed AnnData object', type=str, required=True)
parser.add_argument('-s', '--save_name', help='name of directory to save results', type=str, required=True)
parser.add_argument('-tp', '--timepoints', help='list of timepoints', nargs='+', required=True)