import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_adata", required=True, description="Path to single cell data")
    parser.add_argument('-o', '--output_dir', required=True, description="Path to output directory")
