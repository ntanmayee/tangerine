<p align="center">
    <img src="assets/logo.png" alt="tangerine logo">
</p>

-------
**Tangerine** (Transcription Factor Accessibility Network and Gene Expression Regulation) is a tool to quantify the effect of transcription factor binding on gene regulation.

## Install
Clone the repository and install with poetry by running these commands

```bash
conda config --add channels bioconda
conda config --add channels conda-forge
conda create -n tang python=3.10 scanpy python-igraph leidenalg typer poetry pysam=0.22 bioconda::htseq conda-forge::pyarrow

pip install dash gimmemotifs --no-cache-dir
poetry install --only-root
```

Install the genome of your choice. For example, let's install the `mm10` genome.
```bash
genomepy install mm10 --provider UCSC --annotation
```

## Quick start
Tangerine assumes that your single cell longitudinal data is already available as an 
`AnnData` object, with the `adata.obs` having an additional column indicating the time
of sampling of the cell.

There are two steps to the pipeline - 
1. Process the data to estimate TF-TF and TF-gene regulation
2. Inspect the regulatory relationships using an interactive visualisation

### Preprocess data
```bash
tangerine process \
    -ap path/to/adata \
    -g reference_genome \
    -t "name of time column in adata.obs" \
    -tp "list of time points" \
    -sw "length of scan window in bp" \
    -sp path/to/save/results
```

### Visualise
```bash
tangerine visualise \
    -tp "list of time points" \
    -sp path/to/saved/results
```

## Bug Reports and Suggestions for Improvement
Please raise an issue if you find bugs or if you have any suggestions for improvement.
