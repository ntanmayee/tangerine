import typer
from pathlib import Path
from typing import Optional

from tangerine.utils import print_message
from tangerine.tangerine_pipeline import _tangerine_pipeline

__version__ = "0.1.0"

app = typer.Typer()

def version_callback(value: bool):
    if value:
        typer.echo(f"Tangerine Version: {__version__}")
        raise typer.Exit()
    
@app.callback()
def callback(version: bool = typer.Option(
        None, "--version", callback=version_callback, is_eager=True, help="Display tangerine version"),):
    print_message()
    print(f'Version: {__version__}')

@app.command("process")
def process(
    adata_path: Optional[Path] = typer.Option(None, '--adata_path', '-ap', help='Path to adata object'),
    genome: str = typer.Option(None, '--genome', '-g', help='Reference genome for data. Currently supports `mm10` and `hg19`'),
    bed_file_path: Optional[Path] = typer.Option(None, '--bed_file', '-b', help='Path to local BED file containing gene coordinates (e.g., mm10_genes.bed)'),
    time_var: str = typer.Option('time', '--time', '-t', help='Name of column in adata.obs that denotes time'),
    timepoints: str = typer.Option(None, '--timepoints', '-tp', help='List of timepoints separated by space'),
    scan_width: int = typer.Option(10000, '--scan_width', '-sw', help='Length of window upstream of promoter to search for motifs'),
    metacell_method: str = typer.Option('kmeans', '--metacell_method', '-m', help='Method for generating metacells: "kmeans" or "seacells"'),
    dropout_threshold: int = typer.Option(50, '--dropout_threshold', '-dt', help='Maximum dropout percentage to keep a TF'),
    save_name: Optional[Path] = typer.Option(None, '--save_path', '-sp', help='Path to save results')
):
    print_message()
    
    if not bed_file_path:
        typer.echo("Error: --bed_file is required for offline genomic coordinate mapping.")
        raise typer.Exit(code=1)

    timepoints = timepoints.split()
    
    _tangerine_pipeline(
        'process', 
        adata_path=adata_path,
        timepoints=timepoints,
        genome=genome,
        time_var=time_var,
        scan_width=scan_width,
        save_name=save_name,
        bed_file_path=bed_file_path,
        metacell_method=metacell_method,
        dropout_threshold=dropout_threshold
    )
    
@app.command('visualise')
def visualise(
    timepoints: str = typer.Option(None, '--timepoints', '-tp', help='List of timepoints separated by space'),
    save_name: Optional[Path] = typer.Option(None, '--save_path', '-sp', help='Path to save results')
):
    print_message()
    timepoints = timepoints.split()
    _tangerine_pipeline(
        'visualise',
        timepoints=timepoints,
        save_name=save_name
    )

if __name__ == "__main__":
    app()