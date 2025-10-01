# quick and dirty way to compute ATAC-coverage, needs to be integrated in 
# pipeline later 

if __name__ == "__main__":
    # define constants
    atac_filepaths = {
        '0h': [],
        '2h': []
        '6h': [],
        '18h': [],
        '54h': [],
    }

    # get list of genes - read a networkx result
    