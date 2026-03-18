"""Helper function to run tangerine pipeline from main
"""

from tangerine.preprocess.pipeline import TangerinePipeline 
from tangerine.visualise.app import run_app
from tangerine.preprocess.logger import logger

def _tangerine_pipeline(
        pipeline_step=None, 
        # preprocess arguments
        adata_path=None,
        timepoints=None,
        genome=None,
        time_var=None,
        scan_width=10000,
        save_name=None,
        bed_file_path=None,
        metacell_method='kmeans',
        dropout_threshold=50
        ):
    
    assert timepoints is not None, 'Timepoints must be specified'
    assert save_name is not None, 'Save path must be specified'

    if "process" == pipeline_step:
        # Assertion statements to check required parameters
        assert adata_path is not None, 'Adata object required'
        assert genome is not None, 'Genome must be specified'
        assert time_var is not None, '`time_var` is required. Specify which column in adata.obs denotes time'
        assert bed_file_path is not None, 'A local BED file is required for genomic coordinates'

        adata_path = str(adata_path)
        save_name = str(save_name)
        bed_file_path = str(bed_file_path)

        logger.info(f'Starting data processing using {metacell_method.upper()} metacells...')
        
        # Initialize the new pipeline
        pipeline = TangerinePipeline(
            adata_path=adata_path,
            timepoints=timepoints,
            genome=genome,
            time_var=time_var,
            save_path=save_name,
            scan_width=scan_width,
            metacell_method=metacell_method
        )
        
        # Run the unified execution loop
        pipeline.run(bed_file_path=bed_file_path, dropout_threshold=dropout_threshold)
        
        logger.info('Preprocessing completed successfully.')

    elif "visualise" == pipeline_step:
        save_name = str(save_name)
        logger.info(f'Launching interactive dashboard from {save_name}...')
        run_app(timepoints, save_name)
        
    else:
        logger.error(f"Unknown pipeline step: {pipeline_step}. Use 'process' or 'visualise'.")