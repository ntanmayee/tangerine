import pandas as pd

def download_ucsc_bed(genome='mm10', output_file='mm10_genes.bed'):
    """
    Downloads gene annotations from UCSC and formats them into a standard BED file.
    Supports 'mm10', 'hg19', 'mm39', 'hg38', etc.
    """
    print(f"Downloading {genome} gene annotations from UCSC...")
    
    # The refFlat table contains gene symbols and transcript coordinates
    url = f"http://hgdownload.cse.ucsc.edu/goldenpath/{genome}/database/refFlat.txt.gz"
    
    # refFlat columns: geneName, name, chrom, strand, txStart, txEnd, cdsStart, cdsEnd, exonCount, exonStarts, exonEnds
    df = pd.read_csv(url, sep='\t', header=None, usecols=[0, 2, 3, 4, 5], 
                     names=['symbol', 'chrom', 'strand', 'txStart', 'txEnd'])
    
    # Filter for standard chromosomes (removes unplaced contigs like 'chr1_gl000191_random')
    df = df[df['chrom'].str.match(r'^chr[\dXYMT]+$')]
    
    # A single gene might have multiple transcripts (isoforms). 
    # For basic GRN inference, we usually just take the first listed transcript per gene symbol.
    df = df.drop_duplicates(subset=['symbol'], keep='first')
    
    # Format into standard 6-column BED: chrom, start, end, symbol, score, strand
    bed_df = pd.DataFrame({
        'chrom': df['chrom'],
        'start': df['txStart'],
        'end': df['txEnd'],
        'symbol': df['symbol'],
        'score': 0,          # Dummy score required by standard BED format
        'strand': df['strand']
    })
    
    # Save to file
    bed_df.to_csv(output_file, sep='\t', header=False, index=False)
    print(f"Successfully saved {len(bed_df)} genes to {output_file}")

if __name__ == "__main__":
    # Generate for mouse
    download_ucsc_bed(genome='mm10', output_file='/cluster/gs_lab/tnarendra001/tangerine_data/mm10_genes.bed')
