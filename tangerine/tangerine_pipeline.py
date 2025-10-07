"""Helper function to run tangerine pipeline from main
"""

from tangerine.preprocess.find_motifs import Network 
from tangerine.visualise.app import run_app
from tangerine.preprocess.logger import logger

def _tangerine_pipeline(
        pipeline_step=None, 
        # preprocess arguments
        adata_path=None,
        timepoints=None,
        genome=None,
        time_var=None,
        scan_width=None,
        save_name=None
        ):
    
    assert timepoints is not None, 'Timepoints must be specified'
    assert save_name is not None, 'Save path must be specified'

    if "preprocess" == pipeline_step:
        # assertion statements to check parameters
        assert adata_path is not None, 'Adata object required'
        assert genome is not None, 'Genome must be specified'
        assert time_var is not None, '`time_var` is required. Specify which column in adata.obs denotes time'
        assert scan_width is not None, 'Specify width of scanning window for motifs'
        assert save_name is not None, 'Specify path to save results'

        logger.info('Starting data processing')
        network = Network(adata_path, timepoints, genome, time_var=time_var, scan_width=scan_width)
        network.run_tf_correlation(save_name)
        network.run_all_genes(save_name)
        logger.info('Logging completed.')

    if "visualise" == pipeline_step:
        run_app(timepoints, save_name)
