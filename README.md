<p align="center">
    <img src="assets/logo.png" alt="tangerine logo">
</p>

-------
**Tangerine** (Transcription Factor Accessibility Network and Gene Expression Regulation) is a tool to quantify the effect of transcription factor binding on gene regulation.

## Install
Create a new conda environment
```bash
conda create -n tang
conda activate tang
```

Install required packages
```bash
conda install -c conda-forge scanpy python-igraph leidenalg
pip install scanpy dash gimmemotifs
```

Install the genome of your choice. For example, let's install the `mm10` genome.
```bash
genomepy install mm10 --provider UCSC --annotation
```

## Quick start
### Preprocess data

### Visualise

## Bug Reports and Suggestions for Improvement
Please raise an issue if you find bugs or if you have any suggestions for improvement.
