import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ap', '--adata_path', help='path to processed AnnData object', type=str, required=True)
parser.add_argument('-s', '--save_name', help='name of directory to save results', type=str, required=True)
parser.add_argument('-tp', '--timepoints', help='list of timepoints', nargs='+', required=True)
parser.add_argument('-g', '--genome', help='organism species (`mm10` for Mus musculus and `hg19` for Homo sapiens)', default='mm10')
parser.add_argument('-sw', '--scan_width', help='scan width for promotor region (in bp)', default=5000)
parser.add_argument('-tv', '--time_var', help='name of column in adata.obs that indicates time', default='time')
